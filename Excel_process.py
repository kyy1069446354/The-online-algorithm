import csv

with open(r'C:\Users\KZ\Desktop\2012-01\job.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    starttime=[]
    endtime=[]
    time=[]
    i=0
    for row in reader:
        if row[5]!='0':
            continue
        starttime.append(row[3])
        endtime.append(row[4])
        time.append(int((int(row[4])-int(row[3]))/6000))
    csvfile.close()
    print(time)

with open('C:\\Users\\KZ\\Desktop\\2012-01\\output.csv', "a", newline='') as outputfile:
    writer = csv.writer(outputfile)
    #writer.writerow(["StartTime", "EndTime", "Time"])
    for i in range(len(starttime)):
        writer.writerow([starttime[i],endtime[i],time[i]])
    outputfile.close()

# with open('C:\\Users\\KZ\\Desktop\\data\\output.csv', "r") as csvfile:
#     reader = csv.reader(csvfile)
#     i = 0
#     time = []
#     for row in reader:
#         if i == 0:
#             i = 1
#             continue
#         time.append(int(row[2]))
#     csvfile.close()
#     time.sort()
#     for i in range(305):
#         print(i)
#         print(time.count(i))
