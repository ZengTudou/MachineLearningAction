# k近邻算法
from numpy import *
import operator #运算符模块
from os import listdir # 返回指定目录下的文件名列表


# 创建数据集和标签
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
# kNN分类算法1
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 输入数据集的大小
    # 距离矩阵
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sum(sqDiffMat, axis=1)  # 求行和
    distance = sqDistance ** 0.5
    sortedDistanceIndicies = distance.argsort()  # 将排序后的索引返回
    classCount = {}

    for i in range(k):
        voteLabel = labels[sortedDistanceIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        # 按标签的个数从大到小排序
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    # 样本个数,文件中一行为一个样本
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))  # 3是由于一个样本有3个特征
    classLabelVector = []  # 用来存储标签
    for index, line in enumerate(arrayOLines):
        line = line.strip()  # 去除前后的空格
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[: 3]
        classLabelVector.append(int(listFromLine[-1]))
    return returnMat, classLabelVector

# 将数据归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)  # 取一列中最小的值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = (dataSet - tile(minVals, (m, 1))) / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1  # 测试样本所占的比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 加载数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 归一化数据
    m = normMat.shape[0]
    numTestVecs = int(hoRatio * m)  # 测试样本的个数
    testData = normMat[0:numTestVecs, :]  # 测试样本
    testLabels = datingLabels[0:numTestVecs]
    trainData = normMat[numTestVecs:m, :]  # 训练样本
    trainLabels = datingLabels[numTestVecs:m]
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(testData[i, :], trainData, trainLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifierResult, testLabels[i]))
        if classifierResult != testLabels[i]: errorCount += 1
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not al all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 加载数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inX = array([percentTats,  ffMiles, iceCream])
    result = classify0((inX - minVals)/ranges, normMat, datingLabels, 3)
    print("you will probably like this person:", resultList[result - 1])

# 将2维图像转化为一维数组
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

# 使用k近邻算法来识别手写数字
def handwritingClassTest():
    hwLabels = []  # 加载数据
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)  # 样本个数，一个文件为一个样本
    trainMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 获得文件名，带后缀
        fileStr = fileNameStr.split('.')[0]  # 不带后缀的文件名
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')
    mTest = len(testFileList)
    error = 0.0
    for i in range(mTest):
        fileNameStr = testFileList[i]  # 获得文件名，带后缀
        fileStr = fileNameStr.split('.')[0]  # 不带后缀的文件名
        classNumStr = int(fileStr.split('_')[0])
        inX = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(inX, trainMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is % d"
                %(classifierResult, classNumStr))
        if classifierResult != classNumStr: error += 1.0
    print("the total number of errors is %d" % error)
    print("the total error rate is %f" % (error / mTest))





