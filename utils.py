from __future__ import print_function, division

from _csv import reader as _reader
from datetime import datetime
from json import dumps

# from builtins import input
from time import sleep
from sys import stdout, stderr, version_info
from subprocess import check_output, CalledProcessError, check_call
from os import devnull, path, makedirs
from csv import reader
from re import compile, findall
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from collections import OrderedDict
from typing import List, Callable, Union, Dict, Tuple, Optional
from errno import EEXIST
from six import moves

# Set True to use a known data set for debugging and/or demoing
# Set False to prompt the user for real data
TEST_MODE = False

# The relationship between the LHE content of a cryomodule and the readback from
# the liquid level sensors isn't linear over the full range of the sensors. We
# have chosen to gather all our data with the downstream sensor above 90%. When
# refilling the cryomodule we refill to at least 95%.
MIN_DS_LL = 90
MAX_DS_LL = 95

MIN_US_LL = 66

# Used to reject data where the JT valve wasn't at the correct position
VALVE_POS_TOL = 2

# Used to reject data where the cavity heater wasn't at the correct value
HEATER_TOL = 1.5

# The minimum acceptable run length is ten minutes (600 seconds)
MIN_RUN_DURATION = 600

# We want the liquid level to drop by at least 2.5% during our runs. This isn't
# actually enforced however, unlike the run duration.
TARGET_LL_DIFF = 2.5

# Used to reject data where the cavity gradient wasn't at the correct value
GRAD_TOL = 0.7

# We fetch data from the JLab archiver with a program called MySampler, which
# samples the chosen PVs at a user-specified time interval. Increase to improve
# statistics, decrease to lower the size of the CSV files and speed up
# MySampler data acquisition.
MYSAMPLER_TIME_INTERVAL = 1

# Used in custom input functions
ERROR_MESSAGE = "Please provide valid input"

# This is used to suppress the output of the caput function.
FNULL = open(devnull, "w")

# TODO: Add an INITIAL_CAL_HEAT_LOAD or something like that
# The number of distinct heater settings we're using for cryomodule calibrations
NUM_CAL_STEPS = 12

NUM_LL_POINTS_TO_AVG = 25

CAV_HEATER_RUN_LOAD = 16

CAL_HEATER_DELTA = 1

JT_SEARCH_TIME_RANGE = 24
JT_SEARCH_HOURS_PER_STEP = 0.5
HOURS_NEEDED_FOR_FLATNESS = 1.5


class ValveParams:
    def __init__(self, refValvePos, refHeatLoadDes, refHeatLoadAct):
        self.refValvePos = refValvePos
        self.refHeatLoadDes = refHeatLoadDes
        self.refHeatLoadAct = refHeatLoadAct


class TimeParams:
    def __init__(self, startTime, endTime, timeInterval):
        self.startTime = startTime
        self.endTime = endTime
        self.timeInterval = timeInterval


def isYes(prompt):
    return getStrLim(prompt + " (y/n) ", ["Y", "y", "N", "n"]) in ["y", "Y"]


def getStrLim(prompt, acceptable_strings):
    # type: (str, List[str]) -> str

    response = get_input(prompt, str)

    while response not in acceptable_strings:
        stderr.write(ERROR_MESSAGE + "\n")
        sleep(0.01)
        response = get_input(prompt, str)

    return response


def writeAndFlushStdErr(message):
    # type: (str) -> None
    stderr.write("\n{MSSG}\n".format(MSSG=message))
    stderr.flush()


def writeAndWait(message, timeToWait=0):
    # type: (str, float) -> None
    stdout.write(message)
    stdout.flush()
    sleep(timeToWait)


def get_float_lim(prompt, low_lim, high_lim):
    # type: (str, float, float) -> float
    return getNumericalInput(prompt, low_lim, high_lim, float)


def getNumInputFromLst(prompt, lst, inputType, allowNoResponse=False):
    # type: (str, List[Union[int, float]], Callable, bool) -> Union[float, int]
    response = get_input(prompt, inputType, allowNoResponse)
    while response not in lst:
        # If the user just hits enter, return the first number in the list
        if allowNoResponse and response == "":
            # return lst[0]
            # @pre First option should always be 1
            return 1
        else:
            stderr.write(ERROR_MESSAGE + "\n")
            # Need to pause briefly for some reason to make sure the error message
            # shows up before the next prompt
            sleep(0.01)
            response = get_input(prompt, inputType, allowNoResponse)

    return response


def getNumericalInput(prompt, lowLim, highLim, inputType):
    # type: (str, Union[int, float], Union[int, float], Callable) -> Union[int, float]
    response = get_input(prompt, inputType)

    while response < lowLim or response > highLim:
        stderr.write(ERROR_MESSAGE + "\n")
        sleep(0.01)
        response = get_input(prompt, inputType)

    return response


def get_input(prompt, desired_type, allowNoResponse=False):
    # type: (str, Callable, bool) -> Union[int, float, str]

    response = moves.input(prompt)

    # if allowNoResponse is True the user is permitted to just hit enter in
    # response to the prompt, giving us an empty string regardless of the
    # desired input type
    if allowNoResponse and response == "":
        return response

    try:
        response = desired_type(response)
    except ValueError:
        stderr.write(str(desired_type) + " required\n")
        sleep(0.01)
        return get_input(prompt, desired_type, allowNoResponse)

    return response


def get_int_lim(prompt, low_lim, high_lim):
    # type: (str, int, int) -> int
    return getNumericalInput(prompt, low_lim, high_lim, int)


# PyEpics doesn't work at LERF yet...
# noinspection PyArgumentList
def cagetPV(pv, startIdx=1, attempt=1):
    # type: (str, int, int) -> [str]

    if attempt < 4:
        try:
            out = check_output(["caget", pv, "-n"]).split()[startIdx:]
            if startIdx == 1:
                return out.pop()
            elif startIdx >= 2:
                return out
        except CalledProcessError as _:
            sleep(2)
            print("Retrying caget")
            return cagetPV(pv, startIdx, attempt + 1)

    else:
        raise CalledProcessError("caget failed too many times")


# noinspection PyArgumentList
def caputPV(pv, val, attempt=1):
    # type: (str, str, int) -> int

    if attempt < 4:
        try:
            out = check_call(["caput", pv, val], stdout=FNULL)
            sleep(2)
            return out
        except CalledProcessError:
            sleep(2)
            print("Retrying caput")
            return caputPV(pv, val, attempt + 1)
    else:
        raise CalledProcessError("caput failed too many timeStamps")


def makeTimeFromStr(row, idx):
    # type: (List[str], int) -> datetime
    return datetime.strptime(row[idx], "%m/%d/%y %H:%M:%S")


def getTimeParams(row, indices):
    # type: (List[str], Dict[str, int]) -> TimeParams
    startTime = makeTimeFromStr(row, indices["startIdx"])
    endTime = makeTimeFromStr(row, indices["endIdx"])

    timeIntervalStr = row[indices["timeIntIdx"]]
    timeInterval = (int(timeIntervalStr) if timeIntervalStr
                    else MYSAMPLER_TIME_INTERVAL)

    timeParams = TimeParams(startTime, endTime, timeInterval)

    return timeParams


############################################################################
# getMySamplerData runs a shell command to get archive data. The syntax we're
# using is:
#
#     mySampler -b "%Y-%m-%d %H:%M:%S" -s 1s -n[numPoints] [pv1] ... [pvn]
#
# where the "-b" denotes the start time, "-s 1s" says that the desired time
# step between data points is 1 second, -n[numPoints] tells us how many
# points we want, and [pv1]...[pvn] are the PVs we want archived
#
# Ex:
#     mySampler -b "2019-03-28 14:16" -s 30s -n11 R121PMES R221PMES
#
# @param startTime: datetime object
# @param signals: list of PV strings
############################################################################
def getMySamplerData(startTime, numPoints, signals,
                     timeInt=MYSAMPLER_TIME_INTERVAL):
    # type: (datetime, int, List[str], int) -> Optional[str]
    cmd = (['mySampler', '-b'] + [startTime.strftime("%Y-%m-%d %H:%M:%S")]
           + ['-s', str(timeInt) + 's', '-n' + str(numPoints)]
           + signals)
    try:
        return check_output(cmd)

    except (CalledProcessError, OSError) as e:
        writeAndFlushStdErr("mySampler failed with error: " + str(e) + "\n")
        return None


def getAndParseRawData(startTime, numPoints, signals,
                       timeInt=MYSAMPLER_TIME_INTERVAL, verbose=True):
    # type: (datetime, int, List[str], int, bool) -> Optional[_reader]
    if verbose:
        print("\nGetting data from the archive...\n")
    rawData = getMySamplerData(startTime, numPoints, signals, timeInt)

    if not rawData:
        return None

    else:
        rawDataSplit = rawData.splitlines()
        rows = ["\t".join(rawDataSplit.pop(0).strip().split())]
        rows.extend(list(map(lambda x: reformatDate(x), rawDataSplit)))
        return reader(rows, delimiter='\t')


def reformatDate(row):
    # type: (str) -> unicode
    try:
        # This clusterfuck regex is pretty much just trying to find strings
        # that match %Y-%m-%d %H:%M:%S and making them %Y-%m-%d-%H:%M:%S
        # instead (otherwise the csv parser interprets it as two different
        # columns)
        regex = compile("[0-9]{4}-[0-9]{2}-[0-9]{2}"
                        + " [0-9]{2}:[0-9]{2}:[0-9]{2}")
        res = findall(regex, row)[0].replace(" ", "-")
        reformattedRow = regex.sub(res, row)
        return "\t".join(reformattedRow.strip().split())

    except IndexError:

        writeAndFlushStdErr("Could not reformat date for row: " + str(row)
                            + "\n")
        return "\t".join(row.strip().split())


def genAxis(title, xlabel, ylabel):
    # type: (str, str, str) -> Axes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


# A surprisingly ugly way to pretty print a dictionary
def printOptions(options):
    # type: (Dict[int, str]) -> None
    print(("\n" + dumps(options, indent=4) + "\n")
          .replace('"', '').replace(',', ''))


def addOption(csvRow, lineNum, indices, options):
    # type: (List[str], int, Dict[str, int], Dict[int, str]) -> None
    startTime = makeTimeFromStr(csvRow, indices["startIdx"])
    endTime = makeTimeFromStr(csvRow, indices["endIdx"])
    rate = csvRow[indices["timeIntIdx"]]
    options[lineNum] = ("{START} to {END} ({RATE}s sample interval)"
                        .format(START=startTime, END=endTime,
                                RATE=rate))


def getSelection(duration, suffix, options):
    # type: (float, str, Dict[int, str]) -> int
    # Running a new Q0 measurement or heater calibration is always
    # presented as the last option in the list

    options[max(options) + 1
            if options else 1] = ("Launch new {TYPE} ({DUR} hours)"
                                  .format(TYPE=suffix, DUR=duration))

    # The keys in options are line numbers in a CSV file. We want to present
    # them to the user in a more friendly format, starting from 1 and counting
    # up.
    renumberedOptions = OrderedDict()
    optionMap = {}
    i = 1
    for key in options:
        renumberedOptions[i] = options[key]
        optionMap[i] = key
        i += 1

    printOptions(renumberedOptions)
    formatter = "Please select a {TYPE} option (hit enter for option 1): "
    selection = getNumInputFromLst(formatter.format(TYPE=suffix),
                                   renumberedOptions.keys(), int, True)

    return optionMap[selection]


def drawAndShow():
    # type: () -> None
    plt.draw()
    plt.show()


# Gets raw data from MySampler and does some of the prep work necessary for
# collapsing all of the heater DES and ACT values down into summed values
# (rewrites the CSV header, stores the indices for the heater DES and ACT PVs)
def getDataAndHeaterCols(startTime, numPoints, heaterDesPVs, heaterActPVs,
                         allPVs, timeInt=MYSAMPLER_TIME_INTERVAL, verbose=True):
    # type: (datetime, int, List[str], List[str], List[str], int, bool) -> Optional[List[str], List[int], List[int], List[int], _reader]

    def populateHeaterCols(pvList, buff):
        # type: (List[str], List[float]) -> None
        for heaterPV in pvList:
            buff.append(header.index(heaterPV))

    csvReader = getAndParseRawData(startTime, numPoints, allPVs, timeInt,
                                   verbose)

    if not csvReader:
        return None

    else:

        header = csvReader.next()

        heaterDesCols = []
        populateHeaterCols(heaterDesPVs, heaterDesCols)

        heaterActCols = []
        populateHeaterCols(heaterActPVs, heaterActCols)

        # So that we don't corrupt the indices while we're deleting them
        colsToDelete = sorted(heaterDesCols + heaterActCols, reverse=True)

        for index in colsToDelete:
            del header[index]

        header.append("Electric Heat Load Setpoint")
        header.append("Electric Heat Load Readback")

        return header, heaterActCols, heaterDesCols, colsToDelete, csvReader


def collapseHeaterVals(row, heaterDesCols, heaterActCols):
    # type: (List[str], List[int], List[int]) -> Tuple[Optional[float], Optional[float]]

    heatLoadSetpoint = 0

    for col in heaterDesCols:
        try:
            heatLoadSetpoint += float(row[col])
        except ValueError:
            heatLoadSetpoint = None
            break

    heatLoadAct = 0

    for col in heaterActCols:
        try:
            heatLoadAct += float(row[col])
        except ValueError:
            heatLoadAct = None
            break

    return heatLoadSetpoint, heatLoadAct


def compatibleNext(csvReader):
    # type: (_reader) -> List
    if version_info[0] < 3:
        return csvReader.next()
    else:
        return next(csvReader)


def compatibleMkdirs(filename):
    # type: (str) -> None
    if version_info[0] < 3:
        if not path.exists(path.dirname(filename)):
            try:
                makedirs(path.dirname(filename))
            # Guard against race condition per Stack Overflow
            except OSError as exc:
                if exc.errno != EEXIST:
                    raise
    else:
        makedirs(path.dirname(filename), exist_ok=True)
