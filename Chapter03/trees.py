from math import log
import operator

# 计算一个数据集的熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:   # 先计算出数据集中每个标签的个数
        currentLabel = featVec[-1]  # 取出标签
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 根据属性a和a的取值将数据集合划分
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

# 选择最优划分属性(信息增益最大的属性)
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征的个数
    baseEntropy = calcShannonEnt(dataSet)
    bestFeature = -1
    bestInfoGain = 0.0
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 取出i个特征所有可能的取值
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            newEntropy += (len(subDataSet) / float(len(dataSet))) * \
                          calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(bestInfoGain < infoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 如果所有属性相同，但是类别不同，选择类别最多的类作为最终的标签
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)  # 从大到小排列
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 所有类别都相同
        return classList[0]
    if len(dataSet[0]) == 1:  # 所有特征都用完了
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 找到最优划分属性
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featList = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featList)  # 找到最优划分属性的所有可能取值
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 使用决策树进行分类，返回对应的类标签
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 使用序列化对象存储决策树
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

# 从存储的文件中读取决策树
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)
