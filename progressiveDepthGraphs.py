import statistics as stats
import matplotlib.pyplot as plt
import math
import numpy as np
from os import listdir
import csv

if __name__ == "__main__":
    inputWidths = {"addnist":64,"flowers":24,"goodvbad":48,"rice":5,"mnist":5}
    titles = {
        "addnist":"AddNIST",
        "flowers":"Flowers",
        "goodvbad":"Good Guys-Bad Guys",
        "rice":"Rice",
        "mnist":"MNIST",
    }
    names = ["addnist","flowers","goodvbad","rice","mnist"]
    #names = ["mnist"]
    for name in names:

        fig, axes = plt.subplots(5, 2, figsize=(10, 20))
        axesArr = [None, axes[0, 0], axes[0, 1],
                   axes[1, 0], axes[1, 1],
                   axes[2, 0], axes[2, 1],
                   axes[3, 0], axes[3, 1],
                   axes[4, 0], axes[4, 1]]
        for highlightedDepth in range(1, 11):
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
            #for name in names:
            for file in listdir(f"moreResults/{name}"):                         #CHANGE THIS BACK
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
                    # raise Exception(f"{file} not 10 results")
                    print(f"{file} not 10 results")
                    continue
                trainMean = round(stats.mean(trainAcc),4)
                testMean = round(stats.mean(testAcc),4)
                trainStdev = round(stats.stdev(trainAcc),4)
                testStdev = round(stats.stdev(testAcc),4)
                if depth != highlightedDepth:
                    try:
                        exprs[file] = \
                        ((width / inputWidths[name]) ** ((depth - 1) * inputWidths[name])) * (width ** inputWidths[name])
                        if math.isinf(exprs[file]):
                            del exprs[file]
                            raise OverflowError
                    except OverflowError:
                        print("infinity")
                        print(name)
                        continue
                    trainMeans[file] = trainMean
                    testMeans[file] = testMean
                    trainStdevs[file] = trainStdev
                    testStdevs[file] = testStdev
                    ratios[file] = width/depth
                else:
                    try:
                        highlightedExprs[file] = ((width / inputWidths[name]) ** ((depth - 1) * inputWidths[name])) * (
                                width ** inputWidths[name])
                        if math.isinf(highlightedExprs[file]):
                            del highlightedExprs[file]
                            raise OverflowError
                    except OverflowError:
                        print("infinity")
                        print(name)
                        continue
                    highlightedTrainMeans[file] = trainMean
                    highlightedTestMeans[file] = testMean




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

            verticalLines = {
                "addnist": 283,
                "flowers": 120,
                "goodvbad": 283,
                "rice": 43,
                "mnist": 32,
            }
            ax = axesArr[highlightedDepth]
            ax.set_title(titles[name] + f" Accuracy - Network Depth {highlightedDepth} Highlighted")
            ax.set_xlabel("Log Expressivity")
            ax.set_ylabel("Mean accuracy")
            ax.scatter(exprsPlot, trainMeansPlot, c="lightblue", label="Train")
            ax.scatter(exprsPlot, testMeansPlot, c="tan", label="Test")
            ax.scatter(highlightedExprsPlot, highlightedTrainMeansPlot, c="blue",
                        label=f"Train - Model Depth {highlightedDepth}", s=100)
            ax.scatter(highlightedExprsPlot, highlightedTestMeansPlot, c="orange",
                        label=f"Test - Model Depth {highlightedDepth}", s=100)
            # plt.axvline(verticalLines[name],c="red")
            yMin, yMax = plt.gca().get_ylim()
            xMin, xMax = plt.gca().get_xlim()
            # plt.text(verticalLines[name] + (xMax-xMin)/7  -9, yMax - ((yMax-yMin)/1.4) -0.05, "Log Expressivity: " + str(verticalLines[name]),c="red",fontdict={"fontsize":12})
            ax.legend()
        fig.tight_layout()
        plt.show()

