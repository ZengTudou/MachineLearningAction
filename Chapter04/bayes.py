from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea',
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him',
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute',
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how',
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字， 0代表正常言论
    return postingList, classVec

# 从文档中找到所有不同的单词并建立一个字典
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 给定一个文档和字典，返回文档在字典中的单词序列
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算正例的先验概率
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1> p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'calssified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 使用朴素贝叶斯过滤垃圾邮件
def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []  # 因为之前读入的数据共有50个
    for i in range(10): # 随机取其中10个作为测试集
        randIndex = int (random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    # 训练
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))

    # 测试并计算错误率
    errorCount = 0
    for docIndex in testSet:
        testdata = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(testdata), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is: ", float(errorCount / len(testSet)))

# 使用朴素贝叶斯从个人广告中获取区域倾向
def calcMostFreq(vocabList, fullText): # 返回前30个高频词
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)   # 返回前30个高频词
    for pairW in top30Words:  # remove top 30 words
        if pairW[0] in vocabList: vocabList.remove(pairW[0])  # 这里pariw[0]是因为之前返回的列表的元素是列表
    trainingSet = range(2*minLen); testSet = []           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])


