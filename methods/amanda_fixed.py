import numpy as np
from source import classifiers
from source import metrics
from source import util
from scipy.spatial.distance import euclidean
import sklearn.metrics as met

#此函数是先将数据给分块
def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
    #在这里//表示的意思是整数除法，返回不大于结果的一个最大的整数
             for i in range(wanted_parts) ]

#计算精度
def makeAccuracy(arrAllAcc, arrTrueY):
    arrAcc = []
    ini = 0
    end = ini
    for predicted in arrAllAcc:
        predicted = np.asarray(predicted)
        predicted = predicted.flatten()    #flatten函数返回一个一维数组
        batchSize = len(predicted)
        ini=end
        end=end+batchSize

        yt = arrTrueY[ini:end]
        arrAcc.append(metrics.evaluate(yt, predicted))
        
    return arrAcc


def start(**kwargs):
    dataValues = kwargs["dataValues"]
    dataLabels = kwargs["dataLabels"]
    initialLabeledData = kwargs["initialLabeledData"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    classes = kwargs["classes"]
    K = kwargs["K_variation"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    excludingPercentage = kwargs["excludingPercentage"]
    clfName = kwargs["clfName"]
    densityFunction = kwargs["densityFunction"]
    poolSize = kwargs["poolSize"]
    isBatchMode = kwargs["isBatchMode"]
    
    print("METHOD: {} as classifier and {} as core support extraction with cutting data method".format(clfName, densityFunction))
    usePCA=False
    arrAcc = []
    arrX = []
    arrY = []
    arrUt = []
    arrYt = []
    arrClf = []
    arrPredicted = []
    arrF1 = []
    arrjingdu = []
    arrrecall = []
    initialDataLength = 0
    excludingPercentage = 1-excludingPercentage
    finalDataLength = initialLabeledData #round((initialLabeledDataPerc)*sizeOfBatch)
    reset = True

    # ***** Box 1 *****
    #Initial labeled data    这个数据的预处理中使用的是PCA降维
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    if isBatchMode:
        for t in range(batches):
            #print("passo: ",t)
            initialDataLength=finalDataLength
            finalDataLength=finalDataLength+sizeOfBatch
            #在这里就是数据的分割
            #print(initialDataLength)
            #print(finalDataLength)
            # ***** Box 2 *****在这里应该是对这一块数据进行操作
            Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
            
            # ***** Box 3 *****
          #看这一句
            clf = classifiers.classifier(X, y, K, clfName) #O(nd+kn)

            # for decision boundaries plot    其中append()函数是用于在列表末尾添加新的对象。
            arrClf.append(clf)
            arrX.append(X)
            arrY.append(y)
            arrUt.append(np.array(Ut))
            arrYt.append(yt)
            predicted = clf.predict(Ut)       #只用到了预测结果这一项
            arrPredicted.append(predicted)
            # Evaluating classification
            arrAcc.append(metrics.evaluate(yt, predicted))
            print(t+1,"轮acc",metrics.evaluate(yt, predicted))
            tn, fp, fn, tp = met.confusion_matrix(yt, predicted).ravel()
            recall = tp/(tp+fn)
            f_score = met.f1_score(yt, predicted)
            jingdu = tp/(tp+fp)
            print("recall=",recall)
            print("精度= ",jingdu)
            print("F1score= ",f_score)
            
            # ***** Box 4 *****
            #pdfs from each new points from each class applied on new arrived points
            # allInstances = []
            # allLabels = []
            # if reset == True:
            #     #Considers only the last distribution (time-series like)
            #     pdfsByClass = util.pdfByClass(Ut, predicted, classes, densityFunction)#O(nmd)
            # else:
            #     #Considers the past and actual data (concept-drift like)
            #     allInstances = np.vstack([X, Ut])   #np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
            #     allLabels = np.hstack([y, predicted])  #np.hstack:按水平方向（列顺序）堆叠数组构成一个新的数组
            #     pdfsByClass = util.pdfByClass(allInstances, allLabels, classes, densityFunction)
            #
            # selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)#O(n log(n) c)
            #
            # # ***** Box 6 *****
            # if reset == True:
            #     #Considers only the last distribution (time-series like)
            #     X, y = util.selectedSlicedData(Ut, predicted, selectedIndexes)
            # else:
            #     #Considers the past and actual data (concept-drift like)
            #     X, y = util.selectedSlicedData(allInstances, allLabels, selectedIndexes)#O(n)
    else:
        inst = []
        labels = []
        clf = classifiers.classifier(X, y, K, clfName)
        remainingX , remainingY = util.loadLabeledData(dataValues, dataLabels, finalDataLength, len(dataValues), usePCA)
        reset = False
        
        for Ut, yt in zip(remainingX, remainingY):
            predicted = clf.predict(Ut.reshape(1, -1))[0]
            arrAcc.append(predicted)
            inst.append(Ut)
            labels.append(predicted)

            # for decision boundaries plot
            arrClf.append(clf)
            arrX.append(X)
            arrY.append(y)
            arrUt.append(Ut)
            arrYt.append(yt)
            arrPredicted.append(predicted)
            
            if len(inst) == poolSize:
                inst = np.asarray(inst)
                '''pdfsByClass = util.pdfByClass(inst, labels, classes, densityFunction)
                selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)
                X, y = util.selectedSlicedData(inst, labels, selectedIndexes)
                clf = classifiers.classifier(X, y, K, clfName)
                inst = []
                labels = []'''
                if reset == True:
                    #Considers only the last distribution (time-series like)
                    pdfsByClass = util.pdfByClass(inst, labels, classes, densityFunction)
                else:
                    #Considers the past and actual data (concept-drift like)
                    allInstances = np.vstack([X, inst])
                    allLabels = np.hstack([y, labels])
                    pdfsByClass = util.pdfByClass(allInstances, allLabels, classes, densityFunction)

                selectedIndexes = util.compactingDataDensityBased2(pdfsByClass, excludingPercentage)

                if reset == True:
                    #Considers only the last distribution (time-series like)
                    X, y = util.selectedSlicedData(inst, labels, selectedIndexes)
                else:
                    #Considers the past and actual data (concept-drift like)
                    X, y = util.selectedSlicedData(allInstances, allLabels, selectedIndexes)

                clf = classifiers.classifier(X, y, K, clfName)
                inst = []
                labels = []
            
        arrAcc = split_list(arrAcc, batches)
        arrAcc = makeAccuracy(arrAcc, remainingY)
        arrYt = split_list(arrYt, batches)
        arrPredicted = split_list(arrPredicted, batches)

    # returns accuracy array and last selected points
#    print(arrClf)
    return "AMANDA (Fixed)", arrAcc, X, y, arrX, arrY, arrUt, arrYt, arrClf, arrPredicted, arrF1, arrjingdu, arrrecall