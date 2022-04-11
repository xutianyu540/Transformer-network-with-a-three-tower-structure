import os

import pyedflib
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# DATASET: https://physionet.org/pn6/chbmit/
sampleRate = 256
pathDataSet = ''# path of the dataset
FirstPartPathOutput='' #path where the segments will be saved

# patients = ["01", "02", "03","05","06","07","08","10","23","24"]
patients = ["14"]
channels=18

signalsBlock=None
SecondPartPathOutput=''
legendOfOutput=''
isPreictal=''

_MINUTES_OF_PREICTAL=30


def loadParametersFromFile(filePath):
    global pathDataSet
    global FirstPartPathOutput
    if(os.path.isfile(filePath)):
        with open(filePath, "r") as f:
                line=f.readline()
                if(line.split(":")[0]=="pathDataSet"):
                    pathDataSet=line.split(":")[1].strip()
                line=f.readline()
                if (line.split(":")[0] == "FirstPartPathOutput"):
                    FirstPartPathOutput = line.split(":")[1].strip()

#创建指向索引等于索引的患者文件的指针
def loadSummaryPatient(index):
    f = open(pathDataSet+'/chb'+patients[index]+'/chb'+patients[index]+'-summary.txt', 'r')
    return f

# 将表示时间的字符串转换为日期时间对象
# 和清理不符合小时限制的日期
def getTime(dateInString):
    time=0
    try:
        time = datetime.strptime(dateInString, '%H:%M:%S')
    except ValueError:
        dateInString=" "+dateInString
        if(' 24' in dateInString):
            dateInString = dateInString.replace(' 24', '23')
            time = datetime.strptime(dateInString, '%H:%M:%S')
            time += timedelta(hours=1)
        else:
            dateInString = dateInString.replace(' 25', '23')
            time = datetime.strptime(dateInString, '%H:%M:%S')
            time += timedelta(hours=2)
    return time

#用于表示 Preictal 和 Interictal 数据范围的类
class PreIntData:
    start=0
    end=0
    def __init__(self, s, e):
        self.start=s
        self.end=e
# 用于保存文件数据、开始和结束日期和时间以及相关文件名的类
class FileData:
    start=0
    end=0
    nameFile=""
    def __init__(self, s, e, nF):
        self.start=s
        self.end=e
        self.nameFile=nF

# 将分析的患者的所有有用数据加载到内存中的功能
# 指向已分析患者的摘要文件的指针
def createArrayIntervalData(fSummary):
    preictal = []
    interictal = []
    interictal.append(PreIntData(datetime.min, datetime.max))
    files = []
    firstTime = True
    oldTime = datetime.min  # 相当于日期中的 0
    startTime = 0
    line = fSummary.readline()
    endS = datetime.min

    while (line):
        data = line.split(':')
        if (data[0] == "File Name"):
            nF = data[1].strip()
            s = getTime((fSummary.readline().split(": "))[1].strip())  # 每个edf开始时间
            if (firstTime):
                interictal[0].start = s
                firstTime = False
                startTime = s  # 每个病人开始时间
                endtime = s
            while s < oldTime:  # 如果它每天都在变化，我会在日期上增加 24 小时
                s = s + timedelta(hours=24)
            oldTime = s
            endTimeFile = getTime((fSummary.readline().split(": "))[1].strip())  # 每个edf文件结束时间
            while endTimeFile < oldTime:  # 如果它每天都在变化，我会在日期上增加 24 小时
                endTimeFile = endTimeFile + timedelta(hours=24)
            oldTime = endTimeFile
            files.append(FileData(s, endTimeFile, nF))
            for j in range(0, int((fSummary.readline()).split(':')[1])):
                secSt = int(fSummary.readline().split(': ')[1].split(' ')[0])
                secEn = int(fSummary.readline().split(': ')[1].split(' ')[0])
                ss = s + timedelta(seconds=secSt) - timedelta(minutes=_MINUTES_OF_PREICTAL)  # 发作前_MINUTES_OF_PREICTAL的时间
                if (len(preictal) == 0 or ss > endS):
                    ee = s + timedelta(seconds=secSt)  # 发作的时间
                    preictal.append(PreIntData(ss, ee))  # 发作前30分钟到发作
                endS = s + timedelta(seconds=secEn)  # 发作结束的时间
                ss = s + timedelta(seconds=secSt) - timedelta(hours=4)
                ee = s + timedelta(seconds=secEn) + timedelta(hours=4)
                if (interictal[len(interictal) - 1].start < ss and interictal[len(interictal) - 1].end > ee):
                    interictal[len(interictal) - 1].end = ss
                    interictal.append(PreIntData(ee, datetime.max))
                else:
                    if (interictal[len(interictal) - 1].start < ee):
                        interictal[len(interictal) - 1].start = ee
            if endtime < endTimeFile:
                endtime = endTimeFile
        line = fSummary.readline()
    fSummary.close()
    interictal[len(interictal) - 1].end = endtime

    return preictal, interictal, files

# 加载患者数据（indexPatient）。 数据取自 fileOfData 中指定名称的文件
# 返回包含在文件中的患者数据的 numpy 向量
def loadDataOfPatient(indexPatient, fileOfData):
    f = pyedflib.EdfReader(pathDataSet+'/chb'+patients[indexPatient]+'/'+fileOfData)   # https://pyedflib.readthedocs.io/en/latest/#description
    n = f.signals_in_file
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    sigbufs=cleanData(sigbufs, indexPatient)
    if(patients[indexPatient] in ["15"]):
        #FP1-F7、F7-T7、T7-P7、P7-O1、FP1-F3、F3-C3、C3-P3、P3-O1、FP2-F4、F4-C4、C4-P4、P4-O2、FP2-F8、F8-T8、T8-P8、P8-O2、FZ-CZ、CZ-PZ
        if(fileOfData=='chb15_01.edf'):
            z=[0,1,2,3,5,6,7,8,13,14,15,16,18,19,20,21,10,11]
            sigbufs=sigbufs[z,:]
        else:
            print(fileOfData)
            z=[0,1,2,3,5,6,7,8,14,15,16,17,19,20,21,22,10,11]
            sigbufs=sigbufs[z,:]
    if(patients[indexPatient] in ["11"]):
        #FP1-F7、F7-T7、T7-P7、P7-O1、FP1-F3、F3-C3、C3-P3、P3-O1、FP2-F4、F4-C4、C4-P4、P4-O2、FP2-F8、F8-T8、T8-P8、P8-O2、FZ-CZ、CZ-PZ
        if(fileOfData!='chb11_01.edf'):
            print(fileOfData)
            z=[0,1,2,3,5,6,7,8,13,14,15,16,18,19,20,21,10,11]
            sigbufs=sigbufs[z,:]

    return sigbufs

def saveSignalsOnDisk(signalsBlock,startime,endtime):
    global SecondPartPathOutput
    global FirstPartPathOutput
    global legendOfOutput
    global isPreictal

    if not os.path.exists(FirstPartPathOutput):
        os.makedirs(FirstPartPathOutput)
    if not os.path.exists(FirstPartPathOutput+SecondPartPathOutput):
        os.makedirs(FirstPartPathOutput+SecondPartPathOutput)
    np.save(FirstPartPathOutput+SecondPartPathOutput+'/'+isPreictal+'_'+startime+'-'+endtime, signalsBlock)
    legendOfOutput=legendOfOutput+SecondPartPathOutput+'/'+isPreictal+'_'+startime+'-'+endtime+'.npy\n'

def cleanData(Data, indexPatient):
    if(patients[indexPatient] in ["14","16","17","18","19","20","21","22"]):
        z=[0,1,2,3,5,6,7,8,13,14,15,16,18,19,20,21,10,11]
        Data=Data[z,:]
    return Data

def main():
    global SecondPartPathOutput
    global FirstPartPathOutput
    global legendOfOutput
    global signalsBlock
    global isPreictal
    print("START \n")
    loadParametersFromFile("SEGMENT.txt")
    print("Parameters loaded")
    interictal = []
    for indexPatient in range(0, len(patients)):
        print("Working on patient " + patients[indexPatient])
        legendOfOutput = ""
        allLegend = ""

        SecondPartPathOutput = '/patient' + patients[indexPatient]
        f = loadSummaryPatient(indexPatient)
        preictalInfo, interictalInfo, filesInfo = createArrayIntervalData(f)
        interictalData = np.array([]).reshape(channels, 0)
        indexInterictalSegment = 0
        isPreictal = 'I'
        for fInfo in filesInfo:
            fileS = fInfo.start
            fileE = fInfo.end
            intSegStart = interictalInfo[indexInterictalSegment].start
            intSegEnd = interictalInfo[indexInterictalSegment].end
            while (fileS > intSegEnd and indexInterictalSegment < len(interictalInfo)):
                indexInterictalSegment = indexInterictalSegment + 1
                intSegStart = interictalInfo[indexInterictalSegment].start
                intSegEnd = interictalInfo[indexInterictalSegment].end

            start = 0
            end = 0
            if (not fileE < intSegStart or fileS > intSegEnd):
                if (fileS >= intSegStart):
                    start = 0
                    startime = str(fileS)
                else:
                    start = (intSegStart - fileS).seconds
                    startime = str(intSegStart)
                if (fileE <= intSegEnd):
                    end = None
                    endtime = str(fileE)
                else:
                    end = (intSegEnd - fileS).seconds
                    endtime = str(intSegEnd)
                tmpData = loadDataOfPatient(indexPatient, fInfo.nameFile)
                if (not end == None):
                    end = end * 256
                if (tmpData.shape[0] < channels):
                    print(patients[indexPatient] + "  HA UN NUMERO MINORE DI CANALI")
                else:
                    interictalData = np.concatenate((interictalData, tmpData[0:channels, start * 256:end]), axis=1)
                signalsBlock = interictalData
                saveSignalsOnDisk(signalsBlock, startime, endtime)
                interictalData = np.array([]).reshape(channels, 0)
        legendOfOutput = "INTERICTAL" + "\n" + legendOfOutput
        legendOfOutput = "SEIZURE: " + str(len(preictalInfo)) + "\n" + legendOfOutput
        legendOfOutput = patients[indexPatient] + "\n" + legendOfOutput
        allLegend = legendOfOutput
        print(legendOfOutput)
        legendOfOutput = ''
        print("END create interictal data")


        contSeizure = -1
        isPreictal = 'P'
        for pInfo in preictalInfo:
            contSeizure = contSeizure + 1
            preseg = np.array([]).reshape(channels, 0)
            j = 0
            for j in range(0, len(filesInfo)):
                if (pInfo.start >= filesInfo[j].start and pInfo.start < filesInfo[j].end):
                    break
                if (pInfo.end >= filesInfo[j].start and pInfo.end < filesInfo[j].end):
                    break
            start = (pInfo.start - filesInfo[j].start).seconds
            if (pInfo.start <= filesInfo[j].start):
                start = 0  # 如果 preictal 在文件开头之前开始
            end = None
            tmpData = []
            if (pInfo.end <= filesInfo[j].end):
                end = (pInfo.end - filesInfo[j].start).seconds
                tmpData = loadDataOfPatient(indexPatient, filesInfo[j].nameFile)
                preseg = np.concatenate((preseg, tmpData[0:channels, start * 256:end * 256]), axis=1)
            else:
                tmpData = loadDataOfPatient(indexPatient, filesInfo[j].nameFile)
                preseg = np.concatenate((preseg, tmpData[0:channels, start * 256:]), axis=1)
                end = (pInfo.end - filesInfo[j + 1].start).seconds
                tmpData = loadDataOfPatient(indexPatient, filesInfo[j + 1].nameFile)
                preseg = np.concatenate((preseg, tmpData[0:channels, 0:end * 256]), axis=1)
            signalsBlock = preseg
            startime = str(pInfo.start)
            endtime = str(pInfo.end)
            saveSignalsOnDisk(signalsBlock, startime, endtime)

        allLegend = allLegend + "\n" + "PREICTAL" + "\n" + legendOfOutput
        print(legendOfOutput)


        text_file = open(FirstPartPathOutput + SecondPartPathOutput + "/datamenu.txt", "w")
        text_file.write(allLegend)
        text_file.close()
        print("Legend saved on disk")
        print('\n')


if __name__ == '__main__':
    main()