from pylab import *
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
import csv
import random
import numpy as np
import pandas as pd
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['SimHei']

#大数据分析作业生成器，根据真实数据集生成
def TaskGenerator(n):
    TaskList = []
    AcutualList = []
    #with open('C:\\Users\\KZ\\Desktop\\google cluster\\data\\output0.csv', 'r') as csvfile:
    with open('C:\\Users\\KZ\\Desktop\\2012-01\\output.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        time=[]
        for row in reader:
            #if(int(row[2])>=100 and int(row[2])<=195):
            #if(int(row[2])>=195 and int(row[2])<=545):
            #if(int(row[2])>=545):
            if(int(row[2])>=100):
                time.append(int(row[2]))
        csvfile.close()
        for i in range(n):
            #随机取time中的数，用pop取完删除
            #actual = time.pop(random.randint(0, len(time)-1))
            actual = time[random.randint(0, len(time) - 1)]
            down = actual - random.randint(0, int(0.5*actual))
            if down<0:
                down=1
            up = actual + random.randint(0, int(0.5*actual))
            TaskList.append([down, up, actual])
            AcutualList.append(actual)
    return TaskList,AcutualList

# #任务列表和实际运行时间列表
# TL,AL=TaskGenerator(100)
# print(TL)
# print(AL)

#中间算法，所有作业先串行运行，当空闲机器数达到a时依次并行完成所有任务。
def intermediate(AL,m,a,PF,addPF,SCPF):
    AL.sort()
    RunningTime = 0
    tempAL = []
    flag=AL[a-1]
    RunningTime+=flag
    for i in range(len(AL)):
        if i<a:
            tempAL.append(AL[i])
        else:
            if (AL[i]-flag) == 0:
                tempAL.append(AL[i])
            if (AL[i]-flag) != 0:
                #超出flag的部分计算并行惩罚
                tempAL.append(flag+(AL[i]-flag)*(PF+addPF*(m-1))/m)
                RunningTime+=((tempAL[i]-flag)+AL[i]*SCPF)
    # print(tempAL)
    print('中间算法最终运行时间:'+str(RunningTime))
    return RunningTime
#intermediate(AL,100,10,1,0.05)

#近似最优算法，在知晓所有作业的运行时间和到达的情况下计算出完成这批作业所用的最少时间
def optimal(AL,PF,addPF,SCPF):
    AL.sort()
    RunningTime = 0
    Task=[]
    tempAL=[]
    index=0
    #停止分配空机器的标志
    STOP = False
    #初始化作业列表。Task[0]：作业实际运行时间；Task[1]：作业当前占用哪些机器；Task[2]：作业当前惩罚因子
    for i in AL:
        Task.append([i,[index],PF])
        tempAL.append(i)
        index+=1
    tempMin=min(tempAL)
    #遍历一次就代表出现一次机器空闲
    while sum(tempAL)!=0:
        #运行时间最短的作业跑完，有机器空闲，此时进行方案判断；若有空机器则说明并行额外开销大于并行节省时间
        for i in range(len(tempAL)):
            tempAL[i] -= tempMin
            if tempAL[i]<0:
                tempAL[i]=0
        RunningTime += tempMin
        # print(Task)
        # print(tempAL)
        #每次遍历都分配空机器
        while min(tempAL)==0:
            tempMAX = max(tempAL)
            MaxIndex = tempAL.index(tempMAX)
            #明确有多少台空闲机器
            emptymachine=[]
            for i in range(len(tempAL)):
                if tempAL[i]==0:
                    emptymachine.append(i)
            #全部空闲则说明全部作业运行结束，退出循环
            if len(emptymachine) == len(tempAL):
                break
            #对当前耗时最久的作业进行判断：为其分配更多机器能否缩短总运行时间？
            if tempMAX*len(Task[MaxIndex][1])/Task[MaxIndex][2]*(Task[MaxIndex][2]+addPF*tempAL.count(0))/(tempAL.count(0)+len(Task[MaxIndex][1]))+SCPF*Task[MaxIndex][0] >=tempMAX:
                tempMin = min(filter(lambda x: x > 0, tempAL))
                STOP=True
                break
            mac=[]
            for machine in range(tempAL.count(0)):
                if tempMAX*len(Task[MaxIndex][1])/Task[MaxIndex][2]*(Task[MaxIndex][2]+addPF*(machine+1))/((machine+1)+len(Task[MaxIndex][1]))+SCPF*Task[MaxIndex][0] <tempMAX:
                    mac=machine+1
                    STOP = False
                    break
            #分配机器:惩罚因子上升、机器运行作业更新
            oldtime=tempMAX*len(Task[MaxIndex][1])/Task[MaxIndex][2]
            Task[MaxIndex][1]+=emptymachine[0:mac]
            Task[MaxIndex][2] = PF+addPF*(len(Task[MaxIndex][1])-1)
            for i in Task[MaxIndex][1]:
                Task[i]=Task[MaxIndex]
                tempAL[i]=oldtime*Task[MaxIndex][2]/len(Task[MaxIndex][1])+SCPF*Task[MaxIndex][0]
            # print(Task)
            # print(tempAL)
        if not STOP:
            tempMin = min(tempAL)
    print('最优算法最终运行时间:'+str(RunningTime))
    return RunningTime
#optimal(AL,1,0.05)

#取集合的第4个元素
def fourth(elem):
    return elem[3]
#我的算法，循环分配空闲资源
def online(TL,PF,addPF,SCPF,line):
    RunningTime = 0
    Task = []
    tempAL = []
    #循环列表
    L=[]
    point=-1
    index = 0
    for i in TL:
        if i[0]==i[1]:
            value=(i[1]-line)*(i[0]+i[1])/1
        else:
            value = (i[1]-line)*(i[0]+i[1])/(i[1]-i[0])
        # 每个task是一个五元组：实际单机运行时间、部署机器编码集合、惩罚因子、循环队列排列依据、单机运行时间上限
        Task.append([i[2],[index],PF,value,i[1]])
        index+=1
    #这里的循环队列排列依据是：（上限-阈值）*（上限+下限）/（上限-下限），按照循环队列排序依据重新排序
    Task.sort(key=fourth,reverse=True)
    for i in range(len(Task)):
        tempAL.append(Task[i][0])
        L.append(Task[i])
        Task[i][1]=[i]
    tempMin=min(tempAL)
    #遍历一次就代表出现一次机器空闲
    while sum(tempAL) != 0:
        #运行时间最短的作业跑完，有机器空闲
        #随着作业运行完成，其他作业实际运行时间和最大运行时间都减少，这里由于Task和L中元素引用是相同的，所以Task[i][4]减少L中对应的值也会减少
        for i in range(len(tempAL)):
            tempAL[i] -= tempMin
            Task[i][4] -= tempMin
            if tempAL[i] < 0:
                tempAL[i] = 0
            if Task[i][4] < 0:
                Task[i][4] = 0
        #若有作业的最大运行时间已经小于line阈值，那么把它从L中删除，若删除了指针前的作业则指针相应前移
        for i in L:
            if i[4]<=line:
                if L.index(i)<point:
                    point-=1
                L.remove(i)
        RunningTime += tempMin
        if sum(tempAL) == 0:
            print('我的算法最终运行时间:' + str(RunningTime))
            return RunningTime
        #若L已为空则剩余所有作业都不需要再并行进行，串行运行至结束即可
        if not L:
            RunningTime += max(tempAL)
            print('我的算法最终运行时间:' + str(RunningTime))
            return RunningTime
        # print(Task)
        # print(tempAL)
        # print(L)
        #判断是否需要将上次操作的惩罚时间减去
        PunishorNot=[]
        for i in range(len(TL)):
            PunishorNot.append(0)
        #每次遍历都分配空机器
        while min(tempAL) == 0:
            tempMin = min(tempAL)
            MinIndex = tempAL.index(tempMin)
            #将已经运行完的作业从L中删除掉，若删除了指针前的作业则指针相应前移
            if Task[MinIndex] in L:
                if L.index(Task[MinIndex]) < point:
                    point -= 1
                L.remove(Task[MinIndex])
                # 若L已为空则剩余所有作业都不需要再并行进行，串行运行至结束即可
                if not L:
                    RunningTime += max(tempAL)
                    print('我的算法最终运行时间:' + str(RunningTime))
                    return RunningTime
            #指针在此后移，选定循环队列中的下一个作业
            point+=1
            #指针转一圈后归位
            if point>=len(L)-1:
                point=0
            tempL=L[point]
            MaxIndex=Task.index(tempL)
            tempMAX=tempAL[Task[MaxIndex][1][0]]
            #判断，若选定的作业最大运行时间大于阈值line则为其分配多一台机器
            if Task[MaxIndex][4]>line:
                # 分配机器、惩罚因子上升、每台机器上运行的作业更新
                Task[MaxIndex][1].append(MinIndex)
                Task[MaxIndex][2] += addPF
                if PunishorNot[MaxIndex]==0:
                    Task[MaxIndex][4] = Task[MaxIndex][4] * Task[MaxIndex][2] / len(Task[MaxIndex][1]) + SCPF * Task[MaxIndex][0]
                    PunishorNot[MaxIndex] = 1
                else:
                    Task[MaxIndex][4] = (Task[MaxIndex][4] - SCPF * Task[MaxIndex][0])*(len(Task[MaxIndex][1])-1)/(Task[MaxIndex][2]-addPF)* Task[MaxIndex][2] / len(Task[MaxIndex][1]) + SCPF * Task[MaxIndex][0]
                for i in Task[MaxIndex][1]:
                    Task[i] = Task[MaxIndex]
                # 作业运行时间更新
                tempAL[MinIndex] = tempMAX * Task[MaxIndex][2] / len(Task[MaxIndex][1]) + SCPF * Task[MaxIndex][0]
                tempAL[MaxIndex] = tempMAX * Task[MaxIndex][2] / len(Task[MaxIndex][1]) + SCPF * Task[MaxIndex][0]
                # print(Task)
                # print(tempAL)
                # print(L)
            #若选定的作业最大运行时间不大于阈值line则将其从L中删除掉，若删除了指针前的作业则指针相应前移
            else:
                if L.index(Task[MaxIndex])<point:
                    point-=1
                L.remove(Task[MaxIndex])
                # 若L已为空则剩余所有作业都不需要再并行进行，串行运行至结束即可
                if not L:
                    RunningTime += max(tempAL)
                    print('我的算法最终运行时间:' + str(RunningTime))
                    return RunningTime
            tempMin = min(tempAL)
    return RunningTime
#online(TL,1,0.05,30)

#折线图
def Line_picmaker(line,length):
    im=[]
    im1=[]
    im2=[]
    ol=[]
    opt=[]

    for i in range(length):
        # m = random.randint(10, 20)
        m = random.randint(10, 1000)
        PF = 1
        addPF = (random.randint(1, 500)) / 1000
        SCPF = (random.randint(1, 500)) / 1000

        TL, AL = TaskGenerator(m)
        # with open('m='+str(m)+",SCPF="+str(SCPF)+",addPF="+str(addPF)+"all.txt", "a") as f:
        #     f.write(str(AL)+'\n')
        OPT=optimal(AL,PF,addPF,SCPF)
        OL=online(TL,PF,addPF,SCPF,line)
        IM=intermediate(AL,m,int(0.2*m),PF+SCPF*(m-1),addPF,SCPF)
        IM1=intermediate(AL,m,int(0.5*m),PF+SCPF*(m-1),addPF,SCPF)
        IM2=intermediate(AL,m,int(0.8*m), PF+SCPF*(m - 1),addPF,SCPF)
        opt.append(OPT)
        if(OPT/IM>1):
            im.append(1)
        else:
            im.append(OPT/IM)
        if(OPT/IM1>1):
            im1.append(1)
        else:
            im1.append(OPT/IM1)
        if(OPT/IM2>1):
            im2.append(1)
        else:
            im2.append(OPT/IM2)
        if(OPT/OL>1):
            ol.append(1)
        else:
            ol.append(OPT/OL)
    x = range(length)
    plt.figure(figsize=(10, 3))
    plt.plot(x, im,color='#FF8C00', label='The two-phase algorithm (a=0.2m)',marker='^',markersize='3')
    plt.plot(x, im1,color='#00BFFF', label='The two-phase algorithm (a=0.5m)',marker='v',markersize='3')
    plt.plot(x, im2,color='#808080', label='The two-phase algorithm (a=0.8m)',marker='s',markersize='3')
    plt.plot(x, ol,color='r', label='The online algorithm',marker='.',markersize='3')
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(loc="upper left",fontsize=14,bbox_to_anchor=(-0.011, 1.03))
    # plt.savefig('small.pdf',dpi=600,format='pdf',bbox_inches = 'tight')
    # plt.savefig('big.pdf',dpi=600,format='pdf',bbox_inches = 'tight')
    # plt.savefig('bad1.pdf', dpi=600, format='pdf', bbox_inches='tight')
    # plt.savefig('bad2.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.clf()
    # plt.show()
#Line_picmaker(100, 100)

#条形图
def Bar_picmaker(line,time):
    im=[]
    im1=[]
    im2=[]
    ol=[]
    opt=[]

    m=random.randint(10,1000)
    TL, AL = TaskGenerator(m)
    bar=[]
    for i in range(time):
        PF = 1
        addPF = (random.randint(1, 500)) / 1000
        SCPF = (random.randint(1, 500)) / 1000
        # with open('m='+str(m)+"all.txt", "a") as f:
        #     f.write(str(AL)+'\n')
        OPT=optimal(AL,PF,addPF,SCPF)
        OL=online(TL,PF,addPF,SCPF,line)
        IM=intermediate(AL,m,int(0.2*m),PF+SCPF*(m-1),addPF,SCPF)
        IM1=intermediate(AL,m,int(0.5*m),PF+SCPF*(m-1),addPF,SCPF)
        IM2=intermediate(AL,m,int(0.8*m), PF+SCPF*(m - 1),addPF,SCPF)
        opt.append(OPT)
        ol.append(OL)
        im.append(IM)
        im1.append(IM1)
        im2.append(IM2)
        bar.append(i)
    bar_width = 0.15
    index_opt = np.arange(len(bar))
    index_OL = index_opt + bar_width
    index_IM = index_opt + bar_width*2
    index_IM1 = index_opt + bar_width*3
    index_IM2 = index_opt + bar_width*4
    plt.figure(figsize=(10, 4))
    plt.bar(index_opt, height=opt, width=bar_width, color='k', label='The offline optimal algorithm')
    plt.bar(index_OL, height=ol, width=bar_width, color='r', label='The online algorithm')
    plt.bar(index_IM, height=im, width=bar_width, color='#FF8C00', label='The two-phase algorithm (a=0.2m)')
    plt.bar(index_IM1, height=im1, width=bar_width, color='#00BFFF', label='The two-phase algorithm (a=0.5m)')
    plt.bar(index_IM2, height=im2, width=bar_width, color='#808080', label='The two-phase algorithm (a=0.8m)')
    plt.xticks(index_opt + bar_width*4 / 2, bar)
    plt.yticks(fontsize=14)
    plt.ylabel('Time(second)',fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel('Groups',fontsize=14)
    plt.legend(loc="upper left",fontsize=14,bbox_to_anchor=(-0.011, 1.03))
    plt.savefig('m='+str(m)+'.pdf', dpi=600, format='pdf',bbox_inches = 'tight')
# for i in range(3):
#     Bar_picmaker(100, 15)

#箱线图
def Box_picmaker(line,time):
    im=[]
    im1=[]
    im2=[]
    ol=[]
    opt=[]

    m=random.randint(10,1000)
    PF=1
    addPF=(random.randint(1,500))/1000
    SCPF=(random.randint(1,500))/1000

    for i in range(time):
        TL, AL = TaskGenerator(m)
        # with open('m='+str(m)+",SCPF="+str(SCPF)+",addPF="+str(addPF)+"all.txt", "a") as f:
        #     f.write(str(AL)+'\n')
        OPT=optimal(AL,PF,addPF,SCPF)
        OL=online(TL,PF,addPF,SCPF,line)
        IM=intermediate(AL,m,int(0.2*m),PF+SCPF*(m-1),addPF,SCPF)
        IM1=intermediate(AL,m,int(0.5*m),PF+SCPF*(m-1),addPF,SCPF)
        IM2=intermediate(AL,m,int(0.8*m), PF+SCPF*(m - 1),addPF,SCPF)
        opt.append(OPT)
        if(OPT/IM>1):
            im.append(1)
        else:
            im.append(OPT/IM)
        if(OPT/IM1>1):
            im1.append(1)
        else:
            im1.append(OPT/IM1)
        if(OPT/IM2>1):
            im2.append(1)
        else:
            im2.append(OPT/IM2)
        if(OPT/OL>1):
            ol.append(1)
        else:
            ol.append(OPT/OL)

    data={
        'The online\nalgorithm':ol,
        'Two-phase\nalgorithm\n(a=0.2m)':im,
        'Two-phase\nalgorithm\n(a=0.5m)': im1,
        'Two-phase\nalgorithm\n(a=0.8m)': im2
    }
    df = pd.DataFrame(data)
    df.plot.box()
    plt.grid(linestyle="--", alpha=0.3)
    plt.xticks(fontsize=14)
    plt.savefig('./'+'m='+str(m)+",SCPF="+str(SCPF)+",addPF="+str(addPF)+'all.pdf',dpi=600,format='pdf',bbox_inches = 'tight')
    plt.clf()
    # plt.show()
# for i in range(3):
#     Box_picmaker(100, 1000)

#散点图
def Swarm_picmaker(line,time):
    im=[]
    im1=[]
    im2=[]
    ol=[]
    opt=[]

    for i in range(time):
        m = random.randint(10, 1000)
        PF = 1
        addPF = (random.randint(1, 500)) / 1000
        SCPF = (random.randint(1, 500)) / 1000
        TL, AL = TaskGenerator(m)
        # with open('m='+str(m)+",SCPF="+str(SCPF)+",addPF="+str(addPF)+"all.txt", "a") as f:
        #     f.write(str(AL)+'\n')
        OPT=optimal(AL,PF,addPF,SCPF)
        OL=online(TL,PF,addPF,SCPF,line)
        IM=intermediate(AL,m,int(0.2*m),PF+SCPF*(m-1),addPF,SCPF)
        IM1=intermediate(AL,m,int(0.5*m),PF+SCPF*(m-1),addPF,SCPF)
        IM2=intermediate(AL,m,int(0.8*m), PF+SCPF*(m - 1),addPF,SCPF)
        opt.append(OPT)
        if(OPT/IM>1):
            im.append(1)
        else:
            im.append(OPT/IM)
        if(OPT/IM1>1):
            im1.append(1)
        else:
            im1.append(OPT/IM1)
        if(OPT/IM2>1):
            im2.append(1)
        else:
            im2.append(OPT/IM2)
        if(OPT/OL>1):
            ol.append(1)
        else:
            ol.append(OPT/OL)

    data={
        'The online\nalgorithm':ol,
        'Two-phase\nalgorithm\n(a=0.2m)':im,
        'Two-phase\nalgorithm\n(a=0.5m)': im1,
        'Two-phase\nalgorithm\n(a=0.8m)': im2
    }
    df = pd.DataFrame(data)
    sns.swarmplot(data=df,size=4)
    plt.tick_params(labelsize=14)
    plt.savefig('Totally_random.pdf',dpi=600,format='pdf',bbox_inches = 'tight')
    plt.clf()
    # plt.show()
# Swarm_picmaker(100, 100)

