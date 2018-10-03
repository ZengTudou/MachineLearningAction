# k近邻算法
from numpy import *
import operator #运算符模块


# 创建数据集和标签
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
# kNN分类算法1
def calssify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 输入数据集的大小
    # 距离矩阵
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sum(sqDiffMat, axis=1) # 求行和
    distance = sqDistance ** 0.5
    sortedDistanceIndicies = distance.argsort() # 将排序后的索引返回
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

