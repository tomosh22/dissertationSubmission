import statistics as stats
import matplotlib.pyplot as plt
import math
import numpy as np
from os import listdir
import csv
from sklearn.metrics import r2_score

if __name__ == "__main__":
    inputWidths = {"addnist":64,"flowers":24,"goodvbad":48,"rice":5,"mnist":5}
    titles = {
        "addnist":"AddNIST",
        "flowers":"Flowers",
        "goodvbad":"Good Guys-Bad Guys",
        "rice":"Rice",
        "mnist":"MNIST",
    }
    verticalLines = {
        "addnist": 283,
        "flowers": 120,
        "goodvbad": 283,
        "rice": 43,
        "mnist": 36,
    }
    names = ["addnist","flowers","goodvbad","rice","mnist"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes[1][2].set_visible(False)

    axes[1][0].set_position([0.24, 0.125, 0.228, 0.343])
    axes[1][1].set_position([0.55, 0.125, 0.228, 0.343])
    axes = {"addnist":axes[0,0],"flowers":axes[0,1],"goodvbad":axes[0,2],"rice":axes[1,0],"mnist":axes[1,1]}
    for name in names:
        exprs = {}
        highlightedExprs = {}
        trainMeans = {}
        testMeans = {}
        trainStdevs = {}
        testStdevs = {}
        ratios = {}
        highlightedTrainMeans = {}
        highlightedTestMeans = {}
        #highestDepth = 99
        for file in listdir(f"moreResults/{name}"):
            width, depth = file.replace(".csv", "").split("_")
            width = int(width)
            depth = int(depth)
            if depth > 99:
                continue


            trainAcc = []
            testAcc = []
            fileIn = open(f"moreResults/{name}/{file}")
            reader = csv.DictReader(fileIn)
            entries = [line.replace("\n","").split(",") for line in fileIn.readlines()[1:] if line != '\n']
            fileIn.close()
            for entry in entries:
                trainAcc.append(round(float(entry[0]),4))
                testAcc.append(round(float(entry[1]),4))
            try:
                assert len(trainAcc) == 10 and len(testAcc) == 10
            except:
                #raise Exception(f"{file} not 10 results")
                print(f"{file} not 10 results")
                continue
            trainMean = round(stats.mean(trainAcc),4)
            testMean = round(stats.mean(testAcc),4)
            trainStdev = round(stats.stdev(trainAcc),4)
            testStdev = round(stats.stdev(testAcc),4)
            if True:
                try:
                    exprs[file] = \
                    ((width / inputWidths[name]) ** ((depth - 1) * inputWidths[name])) * (width ** inputWidths[name])
                    if math.isinf(exprs[file]):
                        del exprs[file]
                        raise OverflowError
                except OverflowError:
                    #print("infinity")
                    #print(name)
                    continue
                if (abs(math.log10(exprs[file]) - verticalLines[name]) / verticalLines[name]) * 100.0 < 100:
                    trainMeans[file] = trainMean
                    testMeans[file] = testMean
                    trainStdevs[file] = trainStdev
                    testStdevs[file] = testStdev
                    ratios[file] = depth/width
                else:
                    del exprs[file]
                    continue
            else:
                continue




        trainMeansPlot = np.array([])
        highlightedTrainMeansPlot = np.array([])
        testMeansPlot = np.array([])
        highlightedTestMeansPlot = np.array([])
        trainStdevsPlot = np.array([])
        testStdevsPlot = np.array([])
        exprsPlot = np.array([])
        highlightedExprsPlot = np.array([])
        ratiosPlot = np.array([])
        for x in ratios.values():
            ratiosPlot = np.append(ratiosPlot,x)
        for x in trainMeans.values():
            trainMeansPlot = np.append(trainMeansPlot,x)
        for x in highlightedTrainMeans.values():
            highlightedTrainMeansPlot = np.append(highlightedTrainMeansPlot,x)
        for x in testMeans.values():
            testMeansPlot = np.append(testMeansPlot, x)
        for x in highlightedTestMeans.values():
            highlightedTestMeansPlot = np.append(highlightedTestMeansPlot,x)
        for x in trainStdevs.values():
            trainStdevsPlot = np.append(trainStdevsPlot, x)
        for x in testStdevs.values():
            testStdevsPlot = np.append(testStdevsPlot, x)
        for x in exprs.values():
            exprsPlot = np.append(exprsPlot,math.log10(x))
        for x in highlightedExprs.values():
            highlightedExprsPlot = np.append(highlightedExprsPlot,math.log10(x))


        axes[name].set_title(titles[name]+f" Accuracy")
        axes[name].set_xlabel("Log Expressivity")
        axes[name].set_ylabel("Mean accuracy")
        axes[name].scatter(exprsPlot,trainMeansPlot,c="blue",label="Train")
        axes[name].scatter(exprsPlot,testMeansPlot,c="orange",label="Test")
        #axes[name].scatter(exprsPlot, highlightedTrainMeansPlot, c="blue", label="Train",s=100)
        #axes[name].scatter(exprsPlot, highlightedTestMeansPlot, c="orange", label="Test",s=100)
        axes[name].axvline(verticalLines[name],c="red")
        yMin,yMax = axes[name].get_ylim()
        xMin,xMax = axes[name].get_xlim()
        axes[name].text(verticalLines[name] + (xMax-xMin)/7  -9, yMax - ((yMax-yMin)/1.4) -0.05, "Log Expressivity: " + str(verticalLines[name]),c="red",fontdict={"fontsize":12})
        axes[name].legend()



        # axes[name].set_title(titles[name] + " Accuracy Against Depth:Width")
        # axes[name].set_xlabel("Depth:Width Ratio")
        # axes[name].set_ylabel("Accuracy")
        # axes[name].scatter(ratiosPlot, trainMeansPlot, c="blue", label="Train")
        # axes[name].scatter(ratiosPlot, testMeansPlot, c="orange", label="Test")
        # yMin,yMax = axes[name].get_ylim()
        # xMin,xMax = axes[name].get_xlim()
        # trainRSquared = np.corrcoef(trainMeansPlot,ratiosPlot)[0,1]**2
        # testRSquared = np.corrcoef(testMeansPlot,ratiosPlot)[0,1]**2
        # axes[name].text((xMax-xMin)/1.6, yMax - ((yMax-yMin)/10), "Train R²: " + str(round(trainRSquared,3)),c="red",fontdict={"fontsize":14})
        # axes[name].text((xMax-xMin)/1.6, yMax - ((yMax-yMin)/6), "Test R²: " + str(round(testRSquared,3)),c="red",fontdict={"fontsize":14})
        #
        # axes[name].legend(loc="upper center")
    plt.show()