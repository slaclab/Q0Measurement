from csv import reader
from datetime import datetime
from matplotlib import pyplot as plt
from numpy import mean, std, polyfit
from sys import maxint, stderr

VALVE_LOCKED_POS = 17

# Could probably figure out a way to use numpy arrays if I get the line count
# from the CSV
time = []
unixTime = []
valvePos = []
flowRate = []
heaterPower = []
downstreamLevel = []
upstreamLevel = []

def appendToBuffers(dataBuffers, startIdx, endIdx):
    for (runBuffer, dataBuffer) in dataBuffers:
        runBuffer.append(dataBuffer[startIdx : endIdx])

def parseData(fileName, cryoModule):

    def genHeader(prefix, suffix):
        return prefix + cryoModule + suffix

    with open(fileName) as csvFile:

        csvReader = reader(csvFile)
        header = csvReader.next()

        columnDict = {}

        for buff, col in [(unixTime, "Unix time"),
                          (valvePos, genHeader("CPV:CM0", ":3001:JT:POS_RBV")),
                          (flowRate, "CFICM0312"),
                          (heaterPower, genHeader("CHTR:CM0", ":1155:HV:POWER")),
                          (downstreamLevel, genHeader("CLL:CM0", ":2301:DS:LVL")),
                          (upstreamLevel, genHeader("CLL:CM0", ":2601:US:LVL"))]:
            try:
                columnDict[col] = {"idx": header.index(col), "buffer": buff}

            except ValueError:
                print >> stderr, "Column " + col + " not found in CSV"

        timeIdx = header.index("time")

        for row in csvReader:

            time.append(datetime.strptime(row[timeIdx], "%Y-%m-%d %H:%M:%S"))
                                          
            for col, idxBuffDict in columnDict.iteritems():
                idxBuffDict["buffer"].append(float(row[idxBuffDict["idx"]]))


def getLiquidLevelChange():
    parseData("LL_test_cropped.csv", "2")
    
    runs, timeRuns, heaterVals = populateRuns(heaterPower, downstreamLevel, 66,
                                 1.2)
                                 
    print "Heater Values: " + str(heaterVals)
    adjustForHeaterSettle(heaterVals, runs, timeRuns)
    
    ax1 = genAxis("Liquid Level as a Function of Time", "Unix Time (s)",
              "Downstream Liquid Level (%)")
              
    ax2 = genAxis("Rate of Change of Liquid Level as a Function of Heater Power",
                  "Heater Power (W)", "dLL/dt (%/s)")

    slopes = []

    for idx, run in enumerate(runs):
        m, b = polyfit(timeRuns[idx], run, 1)
        slopes.append(m)

        ax1.plot(timeRuns[idx], run, label=(str(round(m, 6)) + "%/s @ "
                                            + str(heaterVals[idx]) + " W"))

        ax1.plot(timeRuns[idx], [m*x + b for x in timeRuns[idx]])
        
    ax2.plot(heaterVals, slopes, marker="o", linestyle="None")

    m, b = polyfit(heaterVals, slopes, 1)

    ax2.plot(heaterVals, [m*x + b for x in heaterVals],
             label=(str(m)+" %/(s*W)"))

    ax1.legend(loc='lower right')
    ax2.legend(loc='upper right')
    
    plt.show()
    
def getAverage():
    parseData("data_new.csv", "3")
    
    runs, timeRuns, heaterVals = populateRuns(heaterPower, flowRate, 0, maxint)

    print "Heater Values: " + str(heaterVals)
    adjustForHeaterSettle(heaterVals, runs, timeRuns)

    ax = genAxis("Average Flow Rate as a Function of Heater Power", "Time (s)",
                 "Flow Rate")

    for idx, run in enumerate(runs):
        ave = mean(run)

        print "Average: " + str(ave)
        print "Standard Deviation: " + str(std(run))

        ax.plot(timeRuns[idx], run, label=(str(ave) + " @ "
                                           + str(heaterVals[idx]) + " W"))

        ax.plot(timeRuns[idx], [ave for _ in timeRuns[idx]])

    ax.legend(loc="lower left")
    plt.show()


# Sometimes the heater takes a little while to settle, especially after large
# jumps, which renders the points taken during that time worthless
def adjustForHeaterSettle(heaterVals, runs, timeRuns):
    for idx, heaterVal in enumerate(heaterVals):

        # Scaling factor 55 is derived from an observation that an 11W jump
        # leads to about 600 useless points (assuming it scales linearly)
        cutoff = (int(abs(heaterVal - heaterVals[idx - 1]) * 55)
                  if idx > 0 else 0)
        print "cutoff: " + str(cutoff)

        # Adjusting both buffers to keep them "synchronous"
        runs[idx] = runs[idx][cutoff:]
        timeRuns[idx] = timeRuns[idx][cutoff:]

        currVal = heaterVal


def populateRuns(inputBuffer, outputBuffer, levelLimit, valvePosLimit):
    currVal = inputBuffer[0]
    runs = []
    timeRuns = []
    currIdx = 0
    inputVals = []
    
    for idx, val in enumerate(inputBuffer):
            
        # A "break" condition defining the end of a run
        if (val != currVal or upstreamLevel[idx] < levelLimit
            or abs(valvePos[idx] - VALVE_LOCKED_POS) > valvePosLimit):

            # Keeping only those runs with at least 1000 points
            if idx - currIdx > 1000:
                inputVals.append(currVal)
                appendToBuffers([(runs, outputBuffer), (timeRuns, unixTime)],
                                currIdx, idx)
            
            currIdx = idx
            
        currVal = val
    
    if len(inputBuffer) - currIdx > 1000:
        inputVals.append(inputBuffer[len(inputBuffer) - 1])
        appendToBuffers([(runs, outputBuffer), (timeRuns, unixTime)], currIdx,
                        len(inputBuffer))

    return runs, timeRuns, inputVals

def genAxis(title, xlabel, ylabel):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

getLiquidLevelChange()
#getAverage()
