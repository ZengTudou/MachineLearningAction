from Chapter02 import kNN

group, labels = kNN.createDataSet()

# print(group)
# print(labels)
# 第一小节的测试
# print(kNN.calssify0([0, 0], group, labels, 3))
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
# print(datingDataMat[5, :])
# print(datingLabels[5])
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
kNN.datingClassTest()


