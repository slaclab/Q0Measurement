################################################################################
# A utility used to calculate Q0 for a set of cavities in a given cryomodules
# using the change in 2K helium liquid level (LL) per unit time
# Authors: Lisa Zacarias, Ben Ripman
################################################################################

from __future__ import division
from csv import reader, writer
from datetime import datetime
from subprocess import check_output
from re import compile, findall
from os import walk
from os.path import isfile, join, abspath, dirname
from fnmatch import fnmatch
from matplotlib import pyplot as plt
from numpy import polyfit, linspace
from sys import stderr
from scipy.stats import linregress
from json import dumps
from user_input import *
from cryomodule import *

# The LL readings get wonky when the upstream liquid level dips below 66, and
# when the  valve position is +/- 1.2 from our locked position (found
# empirically)
VALVE_POSITION_TOLERANCE = 1.2
UPSTREAM_LL_LOWER_LIMIT = 66

# The minimum run length was also found empirically (it's long enough to ensure
# the runs it detects are usable data and not just noise)
RUN_LENGTH_LOWER_LIMIT = 500

# True if using a known data set for debugging and/or demoing
IS_DEMO = True


def linkBufferToPV(pv, dataBuffer, columnDict, header):
    try:
        columnDict[pv] = {"idx": header.index(pv), "buffer": dataBuffer}
    except ValueError:
        print >> stderr, "Column " + pv + " not found in CSV"


# parseDataFromCSV parses CSV data to populate the given object's data buffers
# @param obj: either a Cryomodule or Cavity object
def parseDataFromCSV(obj):
    columnDict = {}

    with open(obj.dataFileName) as csvFile:

        csvReader = reader(csvFile)
        header = csvReader.next()

        # Figures out the CSV column that has that PV's data and maps it
        for pv, dataBuffer in obj.pvBufferMap.iteritems():
            linkBufferToPV(pv, dataBuffer, columnDict, header)

        if isinstance(obj, Cryomodule):
            # We do the calibration using a cavity heater (usually cavity 1)
            # instead of RF, so we use the heater PV to parse the calibration
            # data using the different heater settings
            heaterPV = obj.cavities[obj.calCavNum].heaterPV
            linkBufferToPV(heaterPV, obj.heatLoadBuffer, columnDict, header)

        timeIdx = header.index("time")
        timeZero = datetime.utcfromtimestamp(0)

        for row in csvReader:
            dt = datetime.strptime(row[timeIdx], "%Y-%m-%d %H:%M:%S")

            # TODO use this to make the plots more human-friendly
            obj.timeBuffer.append(dt)

            # we use the unix time to make the math easier during data
            # processing
            obj.unixTimeBuffer.append((dt - timeZero).total_seconds())

            # Actually parsing the CSV data into the buffers
            for col, idxBuffDict in columnDict.iteritems():
                try:
                    idxBuffDict["buffer"].append(float(row[idxBuffDict["idx"]]))
                except ValueError:
                    print >> stderr, "Could not parse row: " + str(row)


# @param obj: either a Cryomodule or Cavity object
def processData(obj):
    parseDataFromCSV(obj)

    runs, timeRuns, heatLoads = populateRuns(obj)
    adjustForHeaterSettle(heatLoads, runs, timeRuns)

    if IS_DEMO:
        print "Heat Loads: " + str(heatLoads)

        for timeRun in timeRuns:
            print "Duration of run: " + str((timeRun[-1] - timeRun[0]) / 60.0)

    return plotAndFitData(heatLoads, runs, timeRuns, obj)


################################################################################
# plotAndFitData takes three related arrays, plots them, and fits some trend
# lines
#
# heatLoads, runs, and timeRuns are arrays that all have the same size such that
# heatLoads[i] corresponds to runs[i] corresponds to timeRuns[i]
#
# @param heatLoads: an array containing the heat load per data run
# @param runs: an array of arrays, where each runs[i] is a run of LL data for a
#              given heat load
# @param timeRuns: an array of arrays, where each timeRuns[i] is a list of
#                  timestamps that correspond to that run's LL data
# @param obj: Either a Cryomodule or Cavity object
#
# noinspection PyTupleAssignmentBalance
################################################################################
def plotAndFitData(heatLoads, runs, timeRuns, obj):
    isCalibration = isinstance(obj, Cryomodule)

    if isCalibration:
        suffix = " (Cryomodule " + str(obj.cryModNumSLAC) + " Calibration)"
    else:
        suffix = " (Cavity " + str(obj.cavityNumber) + ")"

    ax1 = genAxis("Liquid Level as a Function of Time" + suffix,
                  "Unix Time (s)", "Downstream Liquid Level (%)")

    # A list to hold our all important dLL/dt values!
    slopes = []

    for idx, run in enumerate(runs):
        m, b, r_val, p_val, std_err = linregress(timeRuns[idx], run)

        # A way to diagnose whether or not we had a long enough run of data
        if IS_DEMO:
            print"R^2: " + str(r_val ** 2)

        slopes.append(m)

        ax1.plot(timeRuns[idx], run, label=(str(round(m, 6)) + "%/s @ "
                                            + str(heatLoads[idx])
                                            + "W Electric Load"))

        ax1.plot(timeRuns[idx], [m * x + b for x in timeRuns[idx]])

    if isCalibration:
        ax1.legend(loc='lower right')
        ax2 = genAxis("Rate of Change of Liquid Level as a Function of Heat"
                      " Load", "Heat Load (W)", "dLL/dt (%/s)")

        ax2.plot(heatLoads, slopes, marker="o", linestyle="None",
                 label="Calibration Data")

        m, b = polyfit(heatLoads, slopes, 1)

        ax2.plot(heatLoads, [m * x + b for x in heatLoads],
                 label=(str(m)+" %/(s*W)"))

        ax2.legend(loc='upper right')

        return m, b, ax2, heatLoads

    else:
        ax1.legend(loc='upper right')
        return slopes


# Sometimes the heater takes a little while to settle, especially after large
# jumps, which renders the points taken during that time useless
def adjustForHeaterSettle(heaterVals, runs, timeRuns):
    for idx, heaterVal in enumerate(heaterVals):

        # Scaling factor 27 is derived from an observation that an 11W jump
        # leads to about 300 useless points (assuming it scales linearly)
        cutoff = (int(abs(heaterVal - heaterVals[idx - 1]) * 27)
                  if idx > 0 else 0)

        if IS_DEMO:
            print "cutoff: " + str(cutoff)

        # Adjusting both buffers to keep them "synchronous"
        runs[idx] = runs[idx][cutoff:]
        timeRuns[idx] = timeRuns[idx][cutoff:]


def populateRuns(obj):
    def appendToBuffers(dataBuffers, startIdx, endIdx):
        for (runBuffer, dataBuffer) in dataBuffers:
            runBuffer.append(dataBuffer[startIdx: endIdx])

    runStartIdx = 0

    runs = []
    timeRuns = []
    inputVals = []

    for idx, val in enumerate(obj.heatLoadBuffer):

        prevInputVal = obj.heatLoadBuffer[idx - 1] if idx > 0 else val

        # A "break" condition defining the end of a run
        if (val != prevInputVal
                or obj.upstreamLevelBuffer[idx] < UPSTREAM_LL_LOWER_LIMIT
                or (abs(obj.valvePosBuffer[idx] - obj.refValvePos)
                    > VALVE_POSITION_TOLERANCE)
                or idx == len(obj.heatLoadBuffer) - 1):

            # Keeping only those runs with at least <cutoff> points
            if idx - runStartIdx > RUN_LENGTH_LOWER_LIMIT:
                inputVals.append(prevInputVal - obj.refHeaterVal)
                appendToBuffers([(runs, obj.downstreamLevelBuffer),
                                 (timeRuns, obj.unixTimeBuffer)],
                                runStartIdx, idx)

            runStartIdx = idx

    return runs, timeRuns, inputVals


def genAxis(title, xlabel, ylabel):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def calcQ0(gradient, inputHeatLoad, refGradient=16.0, refHeatLoad=9.6,
           refQ0=2.7E10):
    return refQ0 * (refHeatLoad / inputHeatLoad) * ((gradient / refGradient)**2)


def getArchiveData(startTime, nSecs, signals):
    # startTime & endTime are datetime objects, signals is a list of PV names
    cmd = (['mySampler', '-b'] + [startTime.strftime("%Y-%m-%d %H:%M:%S")]
           + ['-s', '1s', '-n'] + [str(nSecs)] + signals)
    return check_output(cmd)


def reformatDate(row):
    try:
        regex = compile(
            "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}")
        res = findall(regex, row)[0].replace(" ", "-")
        reformattedRow = regex.sub(res, row)
        return "\t".join(reformattedRow.strip().split())
    except IndexError:
        print >> stderr, "Could not reformat date for row: " + str(row)
        return "\t".join(row.strip().split())


################################################################################
# generateCSV is a function that takes either a Cryomodule or Cavity object and
# generates a CSV data file if one doesn't already exist
#
# @param startTime, endTime: datetime objects
# @param object: Cryomodule or Cryomodule.Cavity
################################################################################
def generateCSV(startTime, endTime, obj):
    nSecs = int((endTime - startTime).total_seconds())

    # Define a file name for the CSV we're saving. There are calibration files
    # and q0 measurement files. Both include a time stamp in the format
    # year-month-day--hour-minute. They also indicate the number of data points.
    suffix = startTime.strftime("_%Y-%m-%d--%H-%M_") + str(nSecs) + '.csv'
    cryoModStr = 'CM' + str(obj.cryModNumSLAC)

    if isinstance(obj, Cryomodule.Cavity):
        fileName = ('q0meas_' + cryoModStr + '_cav' + str(obj.cavityNumber)
                    + suffix)

    else:
        fileName = ('calib_' + cryoModStr + suffix)

    if isfile(fileName):
        overwriteFile = get_str('Overwrite previous CSV file (y/n)? ', True,
                                ['y', 'n']) == 'y'
        if not overwriteFile:
            return fileName

    rawData = getArchiveData(startTime, nSecs, obj.getPVs())
    rows = list(map(lambda x: reformatDate(x), rawData.splitlines()))
    csvReader = reader(rows, delimiter='\t')

    with open(fileName, 'wb') as f:
        csvWriter = writer(f, delimiter='\t')
        for row in csvReader:
            csvWriter.writerow(row)

    return fileName


def buildDatetimeFromInput(prompt):
    now = datetime.now()
    year = get_int("Year " + prompt, True, 2019, now.year)

    month = get_int("Month " + prompt, True, 1,
                    now.month if year == now.year else 12)

    day = get_int("Day " + prompt, True, 1,
                  now.day if (year == now.year and month == now.month) else 31)

    hour = get_int("Hour " + prompt, True, 0, 23)
    minute = get_int("Minute " + prompt, True, 0, 59)

    return datetime(year, month, day, hour, minute)


def buildCalibFile(cryomoduleSLAC, cryomoduleLERF, valveLockedPos,
                   refHeaterVal):
    print ("\n***Now we'll start building a calibration file " +
           "- please be patient***\n")

    startTimeCalib = buildDatetimeFromInput("calibration run began: ")
    endTimeCalib = buildDatetimeFromInput("calibration run ended: ")

    cryoModuleObj = Cryomodule(cryomoduleSLAC, cryomoduleLERF, None,
                               valveLockedPos, refHeaterVal)

    cryoModuleObj.dataFileName = generateCSV(startTimeCalib, endTimeCalib,
                                             cryoModuleObj)

    return cryoModuleObj


def findDataFiles(prefix):
    fileDict = {}
    numFiles = 1

    for root, dirs, files in walk(abspath(dirname(__file__))):
        for name in files:
            if fnmatch(name, prefix + "*"):
                fileDict[numFiles] = name
                # fileDict[idx] = join(root, name)
                numFiles += 1

    fileDict[numFiles] = "Generate a new CSV"
    return fileDict


def getQ0Measurements():

    if IS_DEMO:
        refHeaterVal = 1.91
        valveLockedPos = 17.5
        cryomoduleSLAC = 12
        cryomoduleLERF = 2

        cryoModuleObj = Cryomodule(cryomoduleSLAC, cryomoduleLERF,
                                   "calib_CM12_2019-02-25--11-25_18672.csv",
                                   valveLockedPos, refHeaterVal)
        cavities = [2, 4]

    else:
        refHeaterVal = get_float("Reference Heater Value: ", True, 0, 15)
        valveLockedPos = get_float("JT Valve locked position: ", True, 0, 100)

        cryomoduleSLAC = get_int("SLAC Cryomodule Number: ", True, 1, 33)
        cryomoduleLERF = get_int("LERF Cryomodule Number: ", True, 2, 3)

        print "\n---------- CRYOMODULE " + str(cryomoduleSLAC) + " ----------\n"

        calibFiles = findDataFiles("calib_CM" + str(cryomoduleSLAC))

        print "Options for Calibration Data:"

        print "\n" + dumps(calibFiles, indent=4) + "\n"

        option = get_int("Please choose one of the options above: ", True, 1,
                         len(calibFiles))

        if option == len(calibFiles):
            cryoModuleObj = buildCalibFile(cryomoduleSLAC, cryomoduleLERF,
                                           valveLockedPos, refHeaterVal)

        else:
            cryoModuleObj = Cryomodule(cryomoduleSLAC, cryomoduleLERF,
                                       calibFiles[option], valveLockedPos,
                                       refHeaterVal)

        numCavs = get_int("Number of cavities to analyze: ", True, 0, 8)

        cavities = []
        for _ in xrange(numCavs):
            cavity = get_int("Next cavity to analyze: ", True, 1, 8)
            while cavity in cavities:
                cavity = get_int(
                    "Please enter a cavity not previously entered: ",
                    True, 1, 8)
            cavities.append(cavity)

    m, b, ax, calibrationVals = processData(cryoModuleObj)

    for cav in cavities:
        cavityObj = cryoModuleObj.cavities[cav]

        print "\n---------- CAVITY " + str(cav) + " ----------\n"

        q0MeasFiles = findDataFiles("q0meas_CM" + str(cryomoduleSLAC) + "_cav"
                                    + str(cav))

        print "Options for Q0 Meaurement Data:"

        print "\n" + dumps(q0MeasFiles, indent=4) + "\n"

        option = get_int("Please choose one of the options above: ", True, 1,
                         len(q0MeasFiles))

        if option == len(q0MeasFiles):
            print "not implemented"
            return

        else:
            cavityObj.dataFileName = q0MeasFiles[option]

        slopes = processData(cavityObj)

        heaterVals = []

        for dLL in slopes:
            heaterVal = (dLL - b) / m
            heaterVals.append(heaterVal)

        ax.plot(heaterVals, slopes, marker="o", linestyle="None",
                label="Projected Data for Cavity " + str(cav))
        ax.legend(loc="lower left")

        minHeatProjected = min(heaterVals)
        minCalibrationHeat = min(calibrationVals)

        if minHeatProjected < minCalibrationHeat:
            yRange = linspace(minHeatProjected, minCalibrationHeat)
            ax.plot(yRange, [m * i + b for i in yRange])

        maxHeatProjected = max(heaterVals)
        maxCalibrationHeat = max(calibrationVals)

        if maxHeatProjected > maxCalibrationHeat:
            yRange = linspace(maxCalibrationHeat, maxHeatProjected)
            ax.plot(yRange, [m * i + b for i in yRange])

        for heatLoad in heaterVals:
            print "Calculated Heat Load: " + str(heatLoad)
            print "    Q0: " + str(calcQ0(16.05, heatLoad))

    plt.draw()

    # for i in plt.get_fignums():
    #     plt.figure(i)
    #     plt.savefig("figure%d.png" % i)

    plt.show()


if __name__ == "__main__":
    getQ0Measurements()
