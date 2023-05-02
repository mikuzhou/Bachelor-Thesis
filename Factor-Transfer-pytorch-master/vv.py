from matplotlib import pyplot as plt

filename = 'logstu.txt'
filename2='logtea.txt'
filename3='log.txt'
step, stu, tea ,step2,step3,ori= [], [], [],[],[],[]
# 相比open(),with open()不用手动调用close()方法
with open(filename, 'r') as f:
    # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。然后将每个元素中的不同信息提取出来
    lines = f.readlines()
    # i变量，由于这个txt存储时有空行，所以增只读偶数行，主要看txt文件的格式，一般不需要
    # j用于判断读了多少条，step为画图的X轴
    i = 0
    j = 0
    for line in lines:

            temp = line.split(' ')

            t = temp[1].split(':')
            step.append(j)
            j = j + 1


            stu.append(float(temp[4]))
            i = i + 1

with open(filename2, 'r') as f:
    # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。然后将每个元素中的不同信息提取出来
    lines = f.readlines()
    # i变量，由于这个txt存储时有空行，所以增只读偶数行，主要看txt文件的格式，一般不需要
    # j用于判断读了多少条，step为画图的X轴
    i = 0
    j = 0
    for line in lines:

            temp = line.split(' ')

            t = temp[1].split(':')
            step2.append(j)
            j = j + 1


            tea.append(float(temp[4]))
            i = i + 1
with open(filename3, 'r') as f:
    # 将txt中的数据逐行存到列表lines里 lines的每一个元素对应于txt中的一行。然后将每个元素中的不同信息提取出来
    lines = f.readlines()
    # i变量，由于这个txt存储时有空行，所以增只读偶数行，主要看txt文件的格式，一般不需要
    # j用于判断读了多少条，step为画图的X轴
    i = 0
    j = 0
    for line in lines:

            temp = line.split(' ')

            t = temp[1].split(':')
            step3.append(j)
            j = j + 1


            ori.append(float(temp[4]))
            i = i + 1
fig = plt.figure(figsize=(10, 5))  # 创建绘图窗口，并设置窗口大小
# 画第一张图
ax1 = fig.add_subplot(212)  # 将画面分割为2行1列选第一个
ax1.plot(step, stu, 'red', label='stu')  # 画dis-loss的值，颜色红
ax1.legend(loc='upper right')  # 绘制图例，plot()中的label值
ax1.set_xlabel('step')  # 设置X轴名称
ax1.set_ylabel('stu-acc')  # 设置Y轴名称

# 画第二张图
ax2 = fig.add_subplot(212)  # 将画面分割为2行1列选第二个
ax2.plot(step, tea, 'blue', label='tea')  # 画gan-loss的值，颜色蓝
ax2.legend(loc='upper right')  # loc为图例位置，设置在右上方，（右下方为lower right）
ax2.set_xlabel('step')
ax2.set_ylabel('tea-acc')
 # 显示绘制的图
ax3 = fig.add_subplot(212)  # 将画面分割为2行1列选第二个
ax3.plot(step, ori, 'blue', label='kd')  # 画gan-loss的值，颜色蓝
ax3.legend(loc='upper right')  # loc为图例位置，设置在右上方，（右下方为lower right）
ax3.set_xlabel('step')
ax3.set_ylabel('ori-acc')
plt.show()  # 显示绘制的图

plt.figure()
plt.plot(step, stu, 'red', label='stu')
plt.plot(step, tea, 'blue', label='tea')
plt.plot(step, ori, 'black', label='kd')
plt.legend()
plt.show()