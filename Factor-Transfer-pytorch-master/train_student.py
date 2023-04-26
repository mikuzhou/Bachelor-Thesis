import argparse
import json
import time
from datetime import datetime
import warnings
import os


warnings.filterwarnings("ignore")

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import random
from logger import SummaryLogger
import utils
from Models import *


parser = argparse.ArgumentParser(description='Pharaphaser training')
parser.add_argument('--text', default='log.txt', type=str)
parser.add_argument('--exp_name', default='cifar10/FT', type=str)
parser.add_argument('--log_time', default='1', type=str)
parser.add_argument('--lr', default='0.1', type=float)
parser.add_argument('--resume_epoch', default='0', type=int)
parser.add_argument('--epoch', default='163', type=int)
parser.add_argument('--decay_epoch', default=[82, 123], nargs="*", type=int)
parser.add_argument('--w_decay', default='5e-4', type=float)
parser.add_argument('--cu_num', default='0', type=str)
parser.add_argument('--seed', default='1', type=str)
parser.add_argument('--load_pretrained_teacher', default='trained/Teacher.pth', type=str)
parser.add_argument('--load_pretrained_paraphraser_l0', default='trained/Paraphraser_l0.pth', type=str)
parser.add_argument('--load_pretrained_paraphraser_l1', default='trained/Paraphraser_l1.pth', type=str)
parser.add_argument('--load_pretrained_paraphraser_l2', default='trained/Paraphraser_l2.pth', type=str)
parser.add_argument('--save_model', default='ckpt.t7', type=str)
parser.add_argument('--rate', type=float, default=0.5, help='The paraphrase rate k')
parser.add_argument('--beta', type=int, default=500)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parser.parse_args()
print(args)


#### random Seed ####
num = random.randint(1, 10000)
random.seed(num)
torch.manual_seed(num)
#####################


# os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num

#Data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

#Other parameters
# DEVICE = torch.device("cpu")
DEVICE = torch.device("mps")
RESUME_EPOCH = args.resume_epoch
DECAY_EPOCH = args.decay_epoch
DECAY_EPOCH = [ep - RESUME_EPOCH for ep in DECAY_EPOCH]
FINAL_EPOCH = args.epoch
EXPERIMENT_NAME = args.exp_name
W_DECAY = args.w_decay
base_lr = args.lr
RATE = args.rate
BETA = args.beta
map_location = torch.device("cpu")

# Load pretrained models
Teacher = ResNet56()
path = args.load_pretrained_teacher
# state = torch.load(path)
state = torch.load(path, map_location=map_location)
utils.load_checkpoint(Teacher, state)
Teacher.to(DEVICE)

Paraphraser_t2 = Paraphraser(64, int(round(64 * RATE)))
path = args.load_pretrained_paraphraser_l2
# state = torch.load(path)
state = torch.load(path, map_location=map_location)
utils.load_checkpoint(Paraphraser_t2, state)
Paraphraser_t2.to(DEVICE)

Paraphraser_t1 = Paraphraser(32, int(round(64 * RATE)))
path = args.load_pretrained_paraphraser_l1
# state = torch.load(path)
state = torch.load(path, map_location=map_location)
utils.load_checkpoint(Paraphraser_t1, state)
Paraphraser_t1.to(DEVICE)

Paraphraser_t0 = Paraphraser(16, int(round(64 * RATE)))
path = args.load_pretrained_paraphraser_l0
# state = torch.load(path)
state = torch.load(path, map_location=map_location)
utils.load_checkpoint(Paraphraser_t0, state)
Paraphraser_t0.to(DEVICE)

# student models
Student = ResNet18()
Translator_s2 = Translator(64, int(round(64 * RATE)))
Student.to(DEVICE)
Translator_s2.to(DEVICE)
Translator_s1 = Translator(32, int(round(64 * RATE)))
Translator_s0 = Translator(16, int(round(64 * RATE)))
Translator_s1.to(DEVICE)
Translator_s0.to(DEVICE)

# Loss and Optimizer
criterion_CE = nn.CrossEntropyLoss()
criterion = nn.L1Loss()

optimizer = optim.SGD(Student.parameters(), lr=base_lr, momentum=0.9, weight_decay=W_DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
optimizer_module2 = optim.SGD(Translator_s2.parameters(), lr=base_lr, momentum=0.9, weight_decay=W_DECAY)
scheduler_module2 = optim.lr_scheduler.MultiStepLR(optimizer_module2, milestones=DECAY_EPOCH, gamma=0.1)
optimizer_module1 = optim.SGD(Translator_s1.parameters(), lr=base_lr, momentum=0.9, weight_decay=W_DECAY)
scheduler_module1 = optim.lr_scheduler.MultiStepLR(optimizer_module1, milestones=DECAY_EPOCH, gamma=0.1)
optimizer_module0 = optim.SGD(Translator_s0.parameters(), lr=base_lr, momentum=0.9, weight_decay=W_DECAY)
scheduler_module0 = optim.lr_scheduler.MultiStepLR(optimizer_module0, milestones=DECAY_EPOCH, gamma=0.1)
def eval(net):
    loader = testloader
    flag = 'Test'

    epoch_start_time = time.time()
    net.eval()
    val_loss = 0

    correct = 0


    total = 0
    criterion_CE = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs= net(inputs)


        loss = criterion_CE(outputs[3], targets)
        val_loss += loss.item()

        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('%s \t Time Taken: %.2f sec' % (flag, time.time() - epoch_start_time))
    print('Loss: %.3f | Acc net: %.3f%%' % (train_loss / (b_idx + 1), 100. * correct / total))
    return val_loss / (b_idx + 1),  correct / total

def train(teacher,module_t0,module_t1,module_t2,student,module_s0,module_s1,module_s2, epoch):
    epoch_start_time = time.time()
    print('\n EPOCH: %d' % epoch)

    teacher.eval()
    module_t0.eval()
    module_t1.eval()
    module_t2.eval()
    student.train()
    module_s0.train()
    module_s1.train()
    module_s2.train()

    train_loss = 0
    correct = 0
    total = 0

    global optimizer
    global optimizer_module2
    global optimizer_module1
    global optimizer_module0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        optimizer_module2.zero_grad()
        optimizer_module1.zero_grad()
        optimizer_module0.zero_grad()

        # Knowledge transfer with FT loss at the last layer
        ###################################################################################
        teacher_outputs= teacher(inputs)
        student_outputs = student(inputs)

        factor_t2 = module_t2(teacher_outputs[2],1);
        factor_s2 = module_s2(student_outputs[2]);

        factor_t1 = module_t1(teacher_outputs[1],1);
        factor_s1 = module_s1(student_outputs[1]);

        factor_t0 = module_t0(teacher_outputs[0],1);
        factor_s0 = module_s0(student_outputs[0]);

        loss = BETA * 4 / 5 * (criterion(utils.FT(factor_s2), utils.FT(factor_t2.detach()))) \
               + BETA / 10 * (criterion(utils.FT(factor_s1), utils.FT(factor_t1.detach()))) \
               + BETA / 10 * (criterion(utils.FT(factor_s0), utils.FT(factor_t0.detach()))) \
               + criterion_CE(student_outputs[3], targets)
        ###################################################################################
        loss.backward()
        optimizer.step()
        optimizer_module2.step()
        optimizer_module1.step()
        optimizer_module0.step()

        train_loss += loss.item()

        _, predicted = torch.max(student_outputs[3].data, 1)
        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()


        b_idx = batch_idx

    print('Train s1 \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc net: %.3f%%|' % (train_loss / (b_idx + 1), 100. * correct / total))
    return train_loss / (b_idx + 1), correct / total


if __name__ == '__main__':
    time_log = datetime.now().strftime('%m-%d %H:%M')
    if int(args.log_time) :
        folder_name = 'FT_{}'.format(time_log)


    path = os.path.join(EXPERIMENT_NAME, folder_name)
    if not os.path.exists('ckpt/' + path):
        os.makedirs('ckpt/' + path)
    if not os.path.exists('logs/' + path):
        os.makedirs('logs/' + path)

    # Save argparse arguments as logging
    with open('logs/{}/commandline_args.txt'.format(path), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # Instantiate logger
    logger = SummaryLogger(path)


    for epoch in range(RESUME_EPOCH, FINAL_EPOCH+1):
        f = open(os.path.join("logs/" + path, 'log.txt'), "a")

        ### Train ###
        train_loss, acc = train(Teacher, Paraphraser_t0,Paraphraser_t1,Paraphraser_t2,\
                                Student, Translator_s0,Translator_s1,Translator_s2, epoch)
        scheduler.step()
        scheduler_module2.step()
        scheduler_module1.step()
        scheduler_module0.step()

        ### Evaluate  ###
        val_loss, test_acc  = eval(Student)


        f.write('EPOCH {epoch} \t'
                'ACC_net : {acc_net:.4f} \t  \n'.format(epoch=epoch, acc_net=test_acc)
                )
        f.close()

    utils.save_checkpoint({
        'epoch': epoch,
        'state_dict': Student.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, True, 'ckpt/' + path, filename='Model_{}.pth'.format(epoch))

    utils.save_checkpoint({
        'epoch': epoch,
        'state_dict': Translator_s2.state_dict(),
        'optimizer0': optimizer_module0.state_dict(),
        'optimizer1': optimizer_module1.state_dict(),
        'optimizer2': optimizer_module2.state_dict(),
    }, True, 'ckpt/' + path, filename='Translator_{}.pth'.format(epoch))

