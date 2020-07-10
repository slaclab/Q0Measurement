################################################################################
# Utility classes to hold calibration data for a cryomodule, and Q0 measurement
# data for each of that cryomodule's cavities
# Authors: Lisa Zacarias, Ben Ripman
################################################################################

from __future__ import print_function, division
from csv import writer, reader
from decimal import Decimal
from os.path import isfile
from subprocess import CalledProcessError
from time import sleep
from numpy import mean, exp, log10, sqrt, linspace, empty, polyfit, nanmean, nan
from scipy.stats import linregress
from scipy.signal import medfilt
from matplotlib import pyplot as plt
from collections import OrderedDict
from datetime import datetime, timedelta
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Optional, Union
from utils import (writeAndFlushStdErr, MYSAMPLER_TIME_INTERVAL, TEST_MODE,
                   VALVE_POS_TOL, HEATER_TOL, GRAD_TOL,
                   MIN_RUN_DURATION, isYes, writeAndWait, MAX_DS_LL, cagetPV,
                   caputPV, getTimeParams, MIN_DS_LL, getAndParseRawData,
                   genAxis, NUM_CAL_STEPS, NUM_LL_POINTS_TO_AVG,
                   CAV_HEATER_RUN_LOAD, TARGET_LL_DIFF, CAL_HEATER_DELTA,
                   JT_SEARCH_TIME_RANGE, JT_SEARCH_HOURS_PER_STEP,
                   HOURS_NEEDED_FOR_FLATNESS, getDataAndHeaterCols,
                   collapseHeaterVals, ValveParams, TimeParams, compatibleNext,
                   compatibleMkdirs)


RUN_STATUS_MSSG = ("\nWaiting for the LL to drop {DIFF}% "
                   "or below {MIN}%...".format(MIN=MIN_DS_LL, DIFF=TARGET_LL_DIFF))


class Container(object):
    # setting this allows me to create abstract methods and parameters, which
    # are basically things that all inheriting classes MUST implement
    __metaclass__ = ABCMeta

    def __init__(self, cryModNumSLAC, cryModNumJLAB):
        # type: (int, int) -> None
        self.cryModNumSLAC = cryModNumSLAC
        self.cryModNumJLAB = cryModNumJLAB

        self.dsPressurePV = self.addNumToStr("CPT:CM0{CM}:2302:DS:PRESS")
        self.jtModePV = self.addNumToStr("CPV:CM0{CM}:3001:JT:MODE")
        self.jtPosSetpointPV = self.addNumToStr("CPV:CM0{CM}:3001:JT:POS_SETPT")

        # The double curly braces are to trick it into a partial formatting
        # (CM gets replaced first, and {{INFIX}} -> {INFIX} for later)
        lvlFormatStr = self.addNumToStr("CLL:CM0{CM}:{{INFIX}}:{{LOC}}:LVL")

        self.dsLevelPV = lvlFormatStr.format(INFIX="2301", LOC="DS")
        self.usLevelPV = lvlFormatStr.format(INFIX="2601", LOC="US")

        valveLockFormatter = "CPID:CM0{CM}:3001:JT:CV_{SUFF}"
        self.cvMaxPV = self.addNumToStr(valveLockFormatter, "MAX")
        self.cvMinPV = self.addNumToStr(valveLockFormatter, "MIN")
        self.valvePV = self.addNumToStr(valveLockFormatter, "VALUE")

        self.dataSessions = {}

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def idxFile(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def heaterDesPVs(self):
        # type: () -> List[str]
        raise NotImplementedError

    @property
    @abstractmethod
    def heaterActPVs(self):
        # type: () -> List[str]
        raise NotImplementedError

    @property
    @abstractmethod
    def liquidLevelDS(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def totalHeatDes(self):
        raise NotImplementedError

    @abstractmethod
    def waitForTotalHeatDes(self, valveParams):
        # type: (ValveParams) -> None
        raise NotImplementedError

    @abstractmethod
    def walkHeaters(self, perHeaterDelta):
        raise NotImplementedError

    @abstractmethod
    def addDataSessionFromRow(self, row, indices, refHeatLoad, refHeatLoadAct,
                              calibSession=None, refGradVal=None):
        raise NotImplementedError

    @abstractmethod
    def genDataSession(self, timeParams, valveParams, refGradVal=None,
                       calibSession=None):
        raise NotImplementedError

    # Returns a list of the PVs used for this container's data acquisition
    @abstractmethod
    def getPVs(self):
        raise NotImplementedError

    @abstractmethod
    def hash(self, timeParams, slacNum, jlabNum,
             calibSession=None, refGradVal=None):
        raise NotImplementedError

    # getRefValveParams searches over the last timeRange hours for a period
    # when the liquid level was stable and then fetches an averaged JT valve
    # position during that time as well as summed cavity heater DES and ACT
    # values. All three numbers get packaged and returned in a ValveParams
    # object.
    # noinspection PyTupleAssignmentBalance,PyTypeChecker
    def getRefValveParams(self, timeRange=JT_SEARCH_TIME_RANGE):
        # type: (float) -> ValveParams
        
        def halfHourRoundDown(timeToRound):
            # type: (datetime) -> datetime
            newMinute = 0 if timeToRound.minute < 30 else 30
            return datetime(timeToRound.year, timeToRound.month,
                            timeToRound.day, timeToRound.hour, newMinute, 0)
            
        print("\nDetermining required JT Valve position...")

        loopStart = datetime.now()
        searchStart = loopStart - timedelta(hours=HOURS_NEEDED_FOR_FLATNESS)
        searchStart = halfHourRoundDown(searchStart)

        numPoints = int((60 / MYSAMPLER_TIME_INTERVAL)
                        * (HOURS_NEEDED_FOR_FLATNESS * 60))

        while (loopStart - searchStart) <= timedelta(hours=timeRange):

            formatter = "Checking {START} to {END} for liquid level stability."
            searchEnd = searchStart + timedelta(hours=HOURS_NEEDED_FOR_FLATNESS)
            startStr = searchStart.strftime("%m/%d/%y %H:%M:%S")
            endStr = searchEnd.strftime("%m/%d/%y %H:%M:%S")
            print(formatter.format(START=startStr, END=endStr))

            csvReaderLL = getAndParseRawData(searchStart, numPoints,
                                             [self.dsLevelPV], verbose=False)

            compatibleNext(csvReaderLL)
            llVals = []

            for row in csvReaderLL:
                try:
                    llVals.append(float(row.pop()))
                except ValueError:
                    pass

            # Fit a line to the liquid level over the last [numHours] hours
            m, b, _, _, _ = linregress(range(len(llVals)), llVals)

            # If the LL slope is small enough, this may be a good period from
            # which to get a reference valve position & heater params
            if log10(abs(m)) < -5:

                signals = ([self.valvePV] + self.heaterDesPVs
                           + self.heaterActPVs)

                (header, heaterActCols, heaterDesCols, _,
                 csvReader) = getDataAndHeaterCols(searchStart, numPoints,
                                                   self.heaterDesPVs,
                                                   self.heaterActPVs, signals,
                                                   verbose=False)

                valveVals = []
                heaterDesVals = []
                heaterActVals = []
                valveIdx = header.index(self.valvePV)

                for row in csvReader:
                    valveVals.append(float(row[valveIdx]))
                    (heatLoadDes,
                     heatLoadAct) = collapseHeaterVals(row, heaterDesCols,
                                                       heaterActCols)
                    heaterDesVals.append(heatLoadDes)
                    heaterActVals.append(heatLoadAct)

                desValSet = set(heaterDesVals)

                # We only want to use time periods in which there were no
                # changes made to the heater settings
                if len(desValSet) == 1:
                    desPos = round(mean(valveVals), 1)
                    heaterDes = desValSet.pop()
                    heaterAct = mean(heaterActVals)

                    print("Stable period found.")
                    formatter = "{THING} is {VAL}"
                    print(formatter.format(THING="Desired JT valve position",
                                           VAL=desPos))
                    print(formatter.format(THING="Total heater DES setting",
                                           VAL=heaterDes))

                    return ValveParams(desPos, heaterDes, heaterAct)

            searchStart -= timedelta(hours=JT_SEARCH_HOURS_PER_STEP)

        # If we broke out of the while loop without returning anything, that
        # means that the LL hasn't been stable enough recently. Wait a while for
        # it to stabilize and then try again.
        complaint = ("Cryo conditions were not stable enough over the last"
                     " {NUM} hours - determining new JT valve position. Please"
                     " do not adjust the heaters. Allow the PID loop to "
                     "regulate the JT valve position.")
        print(complaint.format(NUM=timeRange))
        
        writeAndWait("\nWaiting 30 minutes for LL to stabilize then "
                     "retrying...")

        start = datetime.now()
        while (datetime.now() - start).total_seconds() < 1800:
            writeAndWait(".", 5)

        # Try again but only search the recent past. We have to manipulate the
        # search range a little bit due to how the search start time is rounded
        # down to the nearest half hour.
        return self.getRefValveParams(HOURS_NEEDED_FOR_FLATNESS + 0.5)

    # We consider the cryo situation to be good when the liquid level is high
    # enough and the JT valve is locked in the correct position
    def waitForCryo(self, refValvePos):
        # type: (float) -> None
        self.waitForLL()
        self.waitForJT(refValvePos)

    def waitForLL(self):
        # type: () -> None
        writeAndWait("\nWaiting for downstream liquid level to be {LL}%..."
                     .format(LL=MAX_DS_LL))

        while (MAX_DS_LL - self.liquidLevelDS) > 0:
            writeAndWait(".", 5)

        writeAndWait(" downstream liquid level at required value.")

    def waitForJT(self, refValvePos):
        # type: (float) -> None

        writeAndWait("\nWaiting for JT Valve to be locked at {POS}..."
                     .format(POS=refValvePos))

        mode = cagetPV(self.jtModePV)

        # One way for the JT valve to be locked in the correct position is for
        # it to be in manual mode and at the desired value
        if mode == "0":
            while float(cagetPV(self.jtPosSetpointPV)) != refValvePos:
                writeAndWait(".", 5)

        # Another way for the JT valve to be locked in the correct position is
        # for it to be automatically regulating and have the upper and lower
        # regulation limits be set to the desired value
        else:

            while float(cagetPV(self.cvMinPV)) != refValvePos:
                writeAndWait(".", 5)

            while float(cagetPV(self.cvMaxPV)) != refValvePos:
                writeAndWait(".", 5)

        writeAndWait(" JT Valve locked.")

    def addNumToStr(self, formatStr, suffix=None):
        # type: (str, Optional[str]) -> str
        if suffix:
            return formatStr.format(CM=self.cryModNumJLAB, SUFF=suffix)
        else:
            return formatStr.format(CM=self.cryModNumJLAB)

    def addDataSession(self, timeParams, valveParams, refGradVal=None,
                       calibSession=None):
        # type: (TimeParams, ValveParams, float, CalibDataSession) -> DataSession

        sessionHash = self.hash(timeParams, self.cryModNumSLAC,
                                self.cryModNumJLAB, calibSession, refGradVal)

        # Only create a new calibration data session if one doesn't already
        # exist with those exact parameters
        if sessionHash not in self.dataSessions:
            session = self.genDataSession(timeParams, valveParams, refGradVal,
                                          calibSession)
            self.dataSessions[sessionHash] = session

        return self.dataSessions[sessionHash]


class Cryomodule(Container):

    def __init__(self, cryModNumSLAC, cryModNumJLAB):
        # type: (int, int) -> None
        super(Cryomodule, self).__init__(cryModNumSLAC, cryModNumJLAB)

        # Give each cryomodule 8 cavities
        cavities = {}

        self._heaterDesPVs = []
        self._heaterActPVs = []

        for i in range(1, 9):
            cav = Cavity(cryMod=self, cavNumber=i)
            cavities[i] = cav
            self._heaterDesPVs.append(cav.heaterDesPV)
            self._heaterActPVs.append(cav.heaterActPV)

        # Using an ordered dictionary so that when we generate the report
        # down the line (iterating over the cavities in a cryomodule), we
        # print the results in order (basic dictionaries aren't guaranteed to
        # be ordered)
        self.cavities = OrderedDict(sorted(cavities.items()))  # type: OrderedDict[int, Cavity]

        self._dsRollingBuff = empty(NUM_LL_POINTS_TO_AVG)
        self._dsRollingBuff[:] = nan

        self._idxFile = ("calibrations/calibrationsCM{CM}.csv"
                         .format(CM=self.cryModNumSLAC))

    @property
    def name(self):
        # type: () -> str
        return "CM{CM}".format(CM=self.cryModNumSLAC)

    @property
    def idxFile(self):
        # type: () -> str

        if not isfile(self._idxFile):
            compatibleMkdirs(self._idxFile)
            with open(self._idxFile, "w+") as f:
                csvWriter = writer(f)
                csvWriter.writerow(["JLAB Number", "Reference Heat Load (Des)",
                                    "Reference Heat Load (Act)",
                                    "JT Valve Position", "Start", "End",
                                    "MySampler Time Interval"])

        return self._idxFile

    @property
    def heaterDesPVs(self):
        # type: () -> List[str]
        return self._heaterDesPVs

    @property
    def heaterActPVs(self):
        # type: () -> List[str]
        return self._heaterActPVs

    @property
    def liquidLevelDS(self):
        # type: () -> float
        try:
            start = datetime.now() - timedelta(seconds=NUM_LL_POINTS_TO_AVG)
            dsValReader = getAndParseRawData(start, NUM_LL_POINTS_TO_AVG,
                                             [self.dsLevelPV],
                                             MYSAMPLER_TIME_INTERVAL,
                                             False)

            # Getting rid of the header
            compatibleNext(dsValReader)

            for row in dsValReader:
                idx = (dsValReader.line_num - 2) % NUM_LL_POINTS_TO_AVG
                self._dsRollingBuff[idx] = float(row[1])

            return nanmean(self._dsRollingBuff)
        except AttributeError:
            return float(cagetPV(self.dsLevelPV))
    
    @property
    def totalHeatDes(self):
        # type: () -> float
        heatDes = 0
        for pv in self.heaterDesPVs:
            heatDes += float(cagetPV(pv))
        return heatDes
    
    def waitForTotalHeatDes(self, valveParams):
        # type: (ValveParams) -> None
        writeAndWait("\nWaiting for total heater setting to be {LOAD} W..."
                     .format(LOAD=valveParams.refHeatLoadDes))
        while self.totalHeatDes != valveParams.refHeatLoadDes:
            writeAndWait(".", 5)

    def hash(self, timeParams, slacNum, jlabNum, calibSession=None,
             refGradVal=None):
        # type: (TimeParams, int, int, CalibDataSession, float) -> int
        return DataSession.hash(timeParams, slacNum, jlabNum)

    # calibSession and refGradVal are unused here, they're just there to match
    # the signature of the overloading method in Cavity (which is why they're in
    # the signature for Container - could probably figure out a way around this)
    def addDataSessionFromRow(self, row, indices, refHeatLoad, refHeatLoadAct,
                              calibSession=None, refGradVal=None):
        # type: (List[str], dict, float, float, CalibDataSession, float) -> CalibDataSession

        timeParams = getTimeParams(row, indices)
        valveParams = ValveParams(float(row[indices["jtIdx"]]),
                                  refHeatLoad, refHeatLoadAct)

        return self.addDataSession(timeParams, valveParams)

    def getPVs(self):
        # type: () -> List[str]
        return ([self.valvePV, self.dsLevelPV, self.usLevelPV]
                + self.heaterDesPVs + self.heaterActPVs)

    def walkHeaters(self, perHeaterDelta):
        # type: (float) -> None

        formatter = "\nWalking CM{NUM} heaters {DIR} by {VAL}"
        dirStr = "up" if perHeaterDelta > 0 else "down"
        formatter = formatter.format(NUM=self.cryModNumSLAC, DIR=dirStr,
                                     VAL=abs(perHeaterDelta))
        print(formatter)

        for heaterSetpointPV in self.heaterDesPVs:
            currVal = float(cagetPV(heaterSetpointPV))
            caputPV(heaterSetpointPV, str(currVal + perHeaterDelta))

        writeAndWait("\nWaiting 5s for cryo to stabilize...\n", 5)

    def genDataSession(self, timeParams, valveParams, refGradVal=None,
                       calibSession=None):
        # type: (TimeParams, ValveParams, float, CalibDataSession) -> CalibDataSession
        return CalibDataSession(self, timeParams, valveParams)

    def runCalibration(self, valveParams=None):
        # type: (ValveParams) -> (CalibDataSession, ValveParams)

        def launchHeaterRun(delta=CAL_HEATER_DELTA):
            # type: (float) -> None
            print("Ramping heaters to the next setting...")

            self.walkHeaters(delta)

            writeAndWait(RUN_STATUS_MSSG)

            startingLevel = self.liquidLevelDS
            avgLevel = startingLevel

            while ((startingLevel - avgLevel) < TARGET_LL_DIFF and (
                    avgLevel > MIN_DS_LL)):
                writeAndWait(".", 10)
                avgLevel = self.liquidLevelDS

            print("\nDone\n")

        # Check whether or not we've already found a good JT position during
        # this program execution
        if not valveParams:
            valveParams = self.getRefValveParams()

        self.waitForCryo(valveParams.refValvePos)
        self.waitForTotalHeatDes(valveParams)

        startTime = datetime.now().replace(microsecond=0)

        launchHeaterRun(8)
        if (self.liquidLevelDS - MIN_DS_LL) < TARGET_LL_DIFF:
            print("Please ask the cryo group to refill to {LL} on the"
                  " downstream sensor".format(LL=MAX_DS_LL))

            self.waitForCryo(valveParams.refValvePos)

        for _ in range(NUM_CAL_STEPS - 1):
            launchHeaterRun()

            if (self.liquidLevelDS - MIN_DS_LL) < TARGET_LL_DIFF:
                print("Please ask the cryo group to refill to {LL} on the"
                      " downstream sensor".format(LL=MAX_DS_LL))

                self.waitForCryo(valveParams.refValvePos)

        # Kinda jank way to avoid waiting for cryo conditions after the final
        # run
        launchHeaterRun()

        endTime = datetime.now().replace(microsecond=0)

        print("\nStart Time: {START}".format(START=startTime))
        print("End Time: {END}".format(END=datetime.now()))

        duration = (datetime.now() - startTime).total_seconds() / 3600
        print("Duration in hours: {DUR}".format(DUR=duration))

        # Walking the heaters back to their starting settings
        # self.walkHeaters(-NUM_CAL_RUNS)

        self.walkHeaters(-((NUM_CAL_STEPS * CAL_HEATER_DELTA) + 1))

        timeParams = TimeParams(startTime, endTime, MYSAMPLER_TIME_INTERVAL)

        dataSession = self.addDataSession(timeParams, valveParams)

        # Record this calibration dataSession's metadata
        with open(self.idxFile, 'a') as f:
            csvWriter = writer(f)
            csvWriter.writerow([self.cryModNumJLAB, valveParams.refHeatLoadDes,
                                valveParams.refHeatLoadAct,
                                valveParams.refValvePos,
                                startTime.strftime("%m/%d/%y %H:%M:%S"),
                                endTime.strftime("%m/%d/%y %H:%M:%S"),
                                MYSAMPLER_TIME_INTERVAL])

        return dataSession, valveParams


class Cavity(Container):

    def __init__(self, cryMod, cavNumber):
        # type: (Cryomodule, int) -> None

        super(Cavity, self).__init__(cryMod.cryModNumSLAC, cryMod.cryModNumJLAB)
        self.parent = cryMod

        self.cavNum = cavNumber
        self._fieldEmissionPVs = None

        heaterDesStr = cryMod.addNumToStr("CHTR:CM0{CM}:1{{CAV}}55:HV:{SUFF}",
                                          "POWER_SETPT")
        heaterActStr = cryMod.addNumToStr("CHTR:CM0{CM}:1{{CAV}}55:HV:{SUFF}",
                                          "POWER")

        self.heaterDesPV = heaterDesStr.format(CAV=cavNumber)
        self.heaterActPV = heaterActStr.format(CAV=cavNumber)

        self._idxFile = ("q0Measurements/q0MeasurementsCM{CM}.csv"
                         .format(CM=self.parent.cryModNumSLAC))

    @property
    def name(self):
        # type: () -> str
        return "Cavity {CAVNUM}".format(CAVNUM=self.cavNum)

    @property
    def idxFile(self):
        # type: () -> str
        if not isfile(self._idxFile):
            compatibleMkdirs(self._idxFile)
            with open(self._idxFile, "w+") as f:
                csvWriter = writer(f)
                csvWriter.writerow(["Cavity", "Gradient",
                                    "JT Valve Position", "Start", "End",
                                    "Reference Heat Load (Des)",
                                    "Reference Heat Load (Act)",
                                    "MySampler Time Interval"])

        return self._idxFile

    @property
    def heaterDesPVs(self):
        # type: () -> List[str]
        return self.parent.heaterDesPVs

    @property
    def heaterActPVs(self):
        # type: () -> List[str]
        return self.parent.heaterActPVs

    @property
    def gradPV(self):
        # type: () -> str
        return self.genAcclPV("GACT")

    @property
    def fieldEmissionPVs(self):
        # type: () -> List[str]
        if not self._fieldEmissionPVs:
            lst = [self.gradPV]
            for suffix in ["ADES", "FWD:PWRMEAN", "REV:PWRMEAN", "CAV:PWRMEAN",
                           "DF"]:
                lst.append(self.genAcclPV(suffix))

            for suffix in ["PEAK", "AVG"]:
                for i in range(1, 3):
                    lst.append("HOM:PWR{IDX}:POWER_{SUFF}".format(IDX=i,
                                                                  SUFF=suffix))

            for i in range(1, 10):
                lst.append("IDRFEL1RAD0{IDX}".format(IDX=i))
                lst.append("IDRFEL2RAD0{IDX}".format(IDX=i))

            for i in range(1, 3):
                lst.append("IDRFEL{IDX}RAD10".format(IDX=i))
                lst.append("IDRFEL{IDX}HVMON".format(IDX=i))

            lst.append(self.dsPressurePV)

            self._fieldEmissionPVs = lst

        return self._fieldEmissionPVs

    @property
    def liquidLevelDS(self):
        # type: () -> float
        return self.parent.liquidLevelDS

    @property
    def totalHeatDes(self):
        # type: () -> float
        return self.parent.totalHeatDes

    def waitForTotalHeatDes(self, valveParams):
        # type: (ValveParams) -> None
        self.parent.waitForTotalHeatDes(valveParams)

    def walkHeater(self, heatDelta):
        # type: (int) -> None

        # negative if we're decrementing heat
        step = 1 if heatDelta > 0 else -1

        formatter = "\nWalking cavity {NUM} heater to {{VAL}}"
        formatter = formatter.format(NUM=self.cavNum)

        for _ in range(abs(heatDelta)):
            currVal = float(cagetPV(self.heaterDesPV))
            newVal = currVal + step
            caputPV(self.heaterDesPV, str(newVal))
            writeAndWait(formatter.format(VAL=newVal), 2)

    # refGradVal and calibSession are required parameters but are nullable to
    # match the signature in Container
    def genDataSession(self, timeParams, valveParams, refGradVal=None,
                       calibSession=None):
        # type: (TimeParams, ValveParams, float, CalibDataSession) -> Q0DataSession
        return Q0DataSession(self, timeParams, valveParams, refGradVal,
                             calibSession)

    def hash(self, timeParams, slacNum, jlabNum, calibSession=None,
             refGradVal=None):
        # type: (TimeParams, int, int, CalibDataSession, float) -> int
        return Q0DataSession.hash(timeParams, slacNum, jlabNum, calibSession,
                                  refGradVal)

    def walkHeaters(self, perHeaterDelta):
        # type: (int) -> None
        self.parent.walkHeaters(perHeaterDelta)

    # calibSession and refGradVal are required parameters for Cavity data
    # sessions, but they're nullable to match the signature in Container
    def addDataSessionFromRow(self, row, indices, refHeatLoad, refHeatLoadAct,
                              calibSession=None, refGradVal=None):
        # type: (List[str], Dict[str, int], float, float, CalibDataSession, float) -> Q0DataSession

        timeParams = getTimeParams(row, indices)
        valveParams = ValveParams(float(row[indices["jtIdx"]]), refHeatLoad,
                                  refHeatLoadAct)

        return self.addDataSession(timeParams, valveParams, refGradVal,
                                   calibSession)

    def genPV(self, formatStr, suffix):
        # type: (str, str) -> str
        return formatStr.format(CM=self.cryModNumJLAB, CAV=self.cavNum,
                                SUFF=suffix)

    def genAcclPV(self, suffix):
        # type: (str) -> str
        return self.genPV("ACCL:L1B:0{CM}{CAV}0:{SUFF}", suffix)

    def getPVs(self):
        # type: () -> List[str]
        return ([self.parent.valvePV, self.parent.dsLevelPV,
                 self.parent.usLevelPV, self.gradPV,
                 self.parent.dsPressurePV] + self.parent.heaterDesPVs
                + self.parent.heaterActPVs)

    # Checks that the parameters associated with acquisition of the cavity RF
    # waveforms are configured properly
    def checkAcqControl(self):
        # type: () -> None
        print("Checking Waveform Data Acquisition Control...")
        for infix in ["CAV", "FWD", "REV"]:
            enablePV = self.genAcclPV(infix + ":ENABLE")
            if float(cagetPV(enablePV)) != 1:
                print("Enabling {INFIX}".format(INFIX=infix))
                caputPV(enablePV, "1")

        suffixValPairs = [("MODE", 1), ("HLDOFF", 0.1), ("STAT_START", 0.065),
                          ("STAT_WIDTH", 0.004), ("DECIM", 255)]

        for suffix, val in suffixValPairs:
            pv = self.genAcclPV("ACQ_" + suffix)
            if float(cagetPV(enablePV)) != val:
                print("Setting {SUFFIX}".format(SUFFIX=suffix))
                caputPV(pv, str(val))

    def setPowerStateSSA(self, turnOn):
        # type: (bool) -> None

        # Using double curly braces to trick it into a partial formatting
        ssaFormatPV = self.genAcclPV("SSA:{SUFFIX}")

        def genPV(suffix):
            return ssaFormatPV.format(SUFFIX=suffix)

        ssaStatusPV = genPV("StatusMsg")

        value = cagetPV(ssaStatusPV)

        if turnOn:
            stateMap = {"desired": "3", "opposite": "2", "pv": "PowerOn"}
        else:
            stateMap = {"desired": "2", "opposite": "3", "pv": "PowerOff"}

        if value != stateMap["desired"]:
            if value == stateMap["opposite"]:
                print("\nSetting SSA power...")
                caputPV(genPV(stateMap["pv"]), "1")

                if cagetPV(ssaStatusPV) != stateMap["desired"]:
                    raise AssertionError("Could not set SSA Power")

            else:
                print("\nResetting SSA...")
                caputPV(genPV("FaultReset"), "1")

                if cagetPV(ssaStatusPV) not in ["2", "3"]:
                    raise AssertionError("Could not reset SSA")

                self.setPowerStateSSA(turnOn)

        print("SSA power set\n")

    ############################################################################
    # Characterize various cavity parameters.
    # * Runs the SSA through its range and constructs a polynomial describing
    #   the relationship between requested SSA output and actual output
    # * Calibrates the cavity's RF probe so that the gradient readback will be
    #   accurate.
    ############################################################################
    def characterize(self):
        # type: () -> None

        def askForVerification(prefix, desAction):
            mssg = "Please {ACTION} manually then press y/Y to continue or n/N to abort"

            if not isYes(prefix + mssg.format(ACTION=desAction)):
                raise AssertionError(
                    "User unsatisfied with characterization - aborting")

        def pushAndWait(suffix):
            caputPV(self.genAcclPV(suffix + "STRT"), "1")

            statusPV = self.genAcclPV(suffix + "STS")

            # 2 is Running
            while cagetPV(statusPV) == "2":
                sleep(1)

            # 0 is Crash
            if cagetPV(statusPV) == "0":
                askForVerification("{CONTROL} crashed. ".format(CONTROL=suffix),
                                   "rerun")


        def checkAndPush(basePV, pushPV, param, tol, newPV=None):

            oldVal = float(cagetPV(self.genAcclPV(basePV)))

            newVal = (float(cagetPV(self.genAcclPV(newPV)))
                      if newPV
                      else float(cagetPV(self.genAcclPV(basePV + "_NEW"))))

            if abs(newVal - oldVal) < tol:
                # pushAndWait(pushPV)
                caputPV(self.genAcclPV(pushPV), "1")

            else:
                askForVerification(
                    "Large difference in {PARAM} ".format(PARAM=param),
                    "inspect and push")

        pushAndWait("SSACAL")

        checkAndPush("SSA:SLOPE", "PUSH_SSASLOPE.PROC", "slopes", 0.15)

        # TODO confirm with Janice what should actually be here
        caputPV(self.genAcclPV("INTLK_RESET_ALL"), "1")
        sleep(2)

        pushAndWait("PROBECAL")

        checkAndPush("QLOADED", "PUSH_QLOADED.PROC", "Loaded Qs", 0.15e7)

        checkAndPush("CAV:SCALER_SEL.B", "PUSH_CAV_SCALE.PROC", "Cavity Scales",
                     0.2, "CAV:CAL_SCALEB_NEW")

    # Switches the cavity to a given operational mode (pulsed, CW, etc.)
    def setModeRF(self, modeDesired):
        # type: (str) -> None

        rfModePV = self.genAcclPV("RFMODECTRL")

        if cagetPV(rfModePV) is not modeDesired:
            caputPV(rfModePV, modeDesired)
            assert cagetPV(rfModePV) == modeDesired, "Unable to set RF mode"

    # Turn the cavity on or off
    def setStateRF(self, turnOn):
        # type: (bool) -> None

        rfStatePV = self.genAcclPV("RFSTATE")
        rfControlPV = self.genAcclPV("RFCTRL")

        rfState = cagetPV(rfStatePV)

        desiredState = ("1" if turnOn else "0")

        if rfState != desiredState:
            print("\nSetting RF State...")
            caputPV(rfControlPV, desiredState)

        print("RF state set\n")

    # Many of the changes made to a cavity don't actually take effect until the
    # go button is pressed
    def pushGoButton(self):
        # type: (Cavity) -> None
        rfStatePV = self.genAcclPV("PULSEONSTRT")
        caputPV(rfStatePV, "1")
        sleep(2)
        if cagetPV(rfStatePV) != "1":
            raise AssertionError("Unable to set RF state")

    # In pulsed mode the cavity has a duty cycle determined by the on time and
    # off time. We want the on time to be 70 ms or else the various cavity
    # parameters calculated from the waveform (e.g. the RF gradient) won't be
    # accurate.
    def checkAndSetOnTime(self):
        # type: () -> None
        print("Checking RF Pulse On Time...")
        onTimePV = self.genAcclPV("PULSE_ONTIME")
        onTime = cagetPV(onTimePV)
        if onTime != "70":
            print("Setting RF Pulse On Time to 70 ms")
            caputPV(onTimePV, "70")
            self.pushGoButton()

    # Ramps the cavity's RF drive (only relevant in pulsed mode) up until the RF
    # gradient is high enough for phasing
    def checkAndSetDrive(self):
        # type: () -> None

        print("Checking drive...")

        drivePV = self.genAcclPV("SEL_ASET")
        currDrive = float(cagetPV(drivePV))

        while (float(cagetPV(self.gradPV)) < 1) or (currDrive < 15):
            print("Increasing drive...")
            driveDes = str(currDrive + 1)

            caputPV(drivePV, driveDes)
            self.pushGoButton()

            currDrive = float(cagetPV(drivePV))

        print("Drive set")

    # Corrects the cavity phasing in pulsed mode based on analysis of the RF
    # waveform. Doesn't currently work if the phase is very far off and the
    # waveform is distorted.
    def phaseCavity(self):
        # type: () -> None

        waveformFormatStr = self.genAcclPV("{INFIX}:AWF")

        def getWaveformPV(infix):
            return waveformFormatStr.format(INFIX=infix)

        # Get rid of trailing zeros - might be more elegant way of doing this
        def trimWaveform(inputWaveform):
            try:
                maxValIdx = inputWaveform.index(max(inputWaveform))
                del inputWaveform[maxValIdx:]

                first = inputWaveform.pop(0)
                while inputWaveform[0] >= first:
                    first = inputWaveform.pop(0)
            except IndexError:
                pass

        def getAndTrimWaveforms():
            res = []

            for suffix in ["REV", "FWD", "CAV"]:
                res.append(cagetPV(getWaveformPV(suffix), startIdx=2))
                res[-1] = list(map(lambda x: float(x), res[-1]))
                trimWaveform(res[-1])

            return res

        def getLine(inputWaveform, lbl):
            return ax.plot(range(len(inputWaveform)), inputWaveform, label=lbl)

        # The waveforms have trailing and leading tails down to zero that would
        # mess with our analysis - have to trim those off.
        revWaveform, fwdWaveform, cavWaveform = getAndTrimWaveforms()

        plt.ion()
        ax = genAxis("Waveforms", "Seconds", "Amplitude")

        ax.set_autoscale_on(True)
        ax.autoscale_view(True, True, True)

        lineRev, = getLine(revWaveform, "Reverse")
        lineCav, = getLine(cavWaveform, "Cavity")
        lineFwd, = getLine(fwdWaveform, "Forward")

        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()

        phasePV = self.genAcclPV("SEL_POFF")

        # When the cavity is properly phased the reverse waveform should dip
        # down very close to zero. Phasing the cavity consists of minimizing
        # that minimum value as we vary the RF phase.
        minVal = min(revWaveform)

        print("Minimum reverse waveform value: {MIN}".format(MIN=minVal))
        step = 1

        while abs(minVal) > 0.5:
            val = float(cagetPV(phasePV))

            print("step: {STEP}".format(STEP=step))
            newVal = val + step
            caputPV(phasePV, str(newVal))

            if float(cagetPV(phasePV)) != newVal:
                writeAndFlushStdErr("Mismatch between desired and actual phase")

            revWaveform, fwdWaveform, cavWaveform = getAndTrimWaveforms()

            lineWaveformPairs = [(lineRev, revWaveform), (lineCav, cavWaveform),
                                 (lineFwd, fwdWaveform)]

            for line, waveform in lineWaveformPairs:
                line.set_data(range(len(waveform)), waveform)

            ax.set_autoscale_on(True)
            ax.autoscale_view(True, True, True)

            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()

            prevMin = minVal
            minVal = min(revWaveform)
            print("Minimum reverse waveform value: {MIN}".format(MIN=minVal))

            # I think this accounts for inflection points? Hopefully the
            # decrease in step size addresses the potential for it to get stuck
            if (prevMin <= 0 < minVal) or (prevMin >= 0 > minVal):
                step *= -0.5

            elif abs(minVal) > abs(prevMin) + 0.01:
                step *= -1

    # Lowers the requested CW amplitude to a safe level where cavities have a
    # very low chance of quenching at turnon
    def lowerAmplitude(self):
        # type: () -> None
        print("Lowering amplitude")
        caputPV(self.genAcclPV("ADES"), "2")

    # Walks the cavity to a given gradient in CW mode with exponential back-off
    # in the step size (steps get smaller each time you cross over the desired
    # gradient until the error is very low)
    def walkToGradient(self, desiredGradient, step=0.5, loopTime=2.5, pv="ADES",
                       getFieldEmissionData=False, gradTol=0.05):
        # type: (float, float, float, str, bool, float) -> None

        amplitudePV = self.genAcclPV(pv)
        gradient = float(cagetPV(self.gradPV))
        diff = gradient - desiredGradient
        prevDiff = diff
        writeAndWait("\nWalking gradient...")

        if getFieldEmissionData:
            formatter = "/u/home/zacarias/Documents/FE/FE_CM{CM}_CAV{CAV}_{DATE}.csv"
            filename = formatter.format(CM=self.cryModNumSLAC,
                                        CAV=self.cavNum,
                                        DATE=datetime.now())
            filename = filename.replace(" ", "_")
            filename = filename.replace(":", "-")

            compatibleMkdirs(filename)
            with open(filename, "w+") as f:
                csvWriter = writer(f)
                csvWriter.writerow(["time"] + self.fieldEmissionPVs)

        while abs(diff) > gradTol:

            loopStartTime = datetime.now()

            writeAndWait(".")

            currAmp = float(cagetPV(amplitudePV))
            diff = gradient - desiredGradient
            mult = 1 if (diff <= 0) else -1

            overshot = (prevDiff >= 0 > diff) or (prevDiff <= 0 < diff)

            # This only works if we're in SEL mode; in pulsed mode the scaling
            # is messed up because a 1% change in the drive doesn't correspond
            # to a 1 MV/m change in the gradient
            if (abs(diff) < (2 * step) or overshot) and (step > gradTol):
                step *= 0.5

            prevDiff = diff

            caputPV(amplitudePV, str(currAmp + mult * step))

            if pv == "SEL_ASET":
                self.pushGoButton()

            gradient = self.checkForQuench(gradient)

            while (datetime.now() - loopStartTime).total_seconds() < loopTime:

                gradient = self.checkForQuench(gradient)

                if getFieldEmissionData:

                    with open(filename, "a") as f:
                        csvWriter = writer(f)
                        row = [datetime.now()]

                        for fieldEmissionPV in self.fieldEmissionPVs:
                            row.append(cagetPV(fieldEmissionPV))

                        csvWriter.writerow(row)

        print("\nGradient at desired value")

    def checkForQuench(self, oldGradient):
        # type: (float) -> float
        newGradient = float(cagetPV(self.gradPV))

        if newGradient < (oldGradient * 0.9):
            formatStr = "Detected a quench at {GRAD} - aborting"

            bypassPV = self.genAcclPV("QUENCH_BYP")

            # If the EPICs quench detection is disabled and we see a quench,
            # shut the cavity down
            if cagetPV(bypassPV) == "1":
                raise AssertionError(formatStr.format(GRAD=oldGradient))
            # If the EPICs quench detection is enabled just print a warning
            # message
            else:
                print(formatStr.format(GRAD=oldGradient))

        return newGradient

    # When cavities are turned on in CW mode they slowly heat up, which causes
    # the gradient to drop over time. This function holds the gradient at the
    # requested value during the Q0 run.
    def holdGradient(self, desiredGradient, minLL=MIN_DS_LL, gradTol=0.01):
        # type: (float, float, float) -> datetime

        amplitudePV = self.genAcclPV("ADES")

        startTime = datetime.now()

        step = 0.01
        prevDiff = float(cagetPV(self.gradPV)) - desiredGradient

        print("\nStart time: {START}".format(START=startTime))

        writeAndWait(
            "\nWaiting for the LL to drop {DIFF}% or below {MIN}%...".format(
                MIN=minLL, DIFF=TARGET_LL_DIFF))

        gradient = float(cagetPV(self.gradPV))

        startingLevel = self.liquidLevelDS
        avgLevel = startingLevel

        # TODO figure out how to squish this with FE measurements
        while ((startingLevel - avgLevel) < TARGET_LL_DIFF
               and (avgLevel > minLL)):

            currAmp = float(cagetPV(amplitudePV))
            gradient = self.checkForQuench(gradient)
            diff = gradient - desiredGradient

            mult = 1 if (diff <= 0) else -1

            overshot = (prevDiff >= 0 > diff) or (prevDiff <= 0 < diff)

            # This only works if we're in SEL mode; in pulsed mode the scaling
            # is messed up because a 1% change in the drive doesn't correspond
            # to a 1 MV/m change in the gradient
            if abs(diff) < gradTol:
                pass
            elif (abs(diff) < (2 * step) or overshot) and (step > gradTol):
                step *= 0.5
            else:
                step *= 1.5

            caputPV(amplitudePV, str(currAmp + mult * step))

            prevDiff = diff

            writeAndWait(".", 5)
            avgLevel = self.liquidLevelDS

        print("\nEnd Time: {END}".format(END=datetime.now()))
        duration = (datetime.now() - startTime).total_seconds() / 3600
        print("Duration in hours: {DUR}".format(DUR=duration))
        return startTime

    def powerDown(self):
        # type: (Cavity) -> None
        try:
            print("\nPowering down...")
            self.setStateRF(False)
            self.setPowerStateSSA(False)
            caputPV(self.genAcclPV("SEL_ASET"), "15")
            self.lowerAmplitude()
        except(CalledProcessError, IndexError, OSError,
               ValueError, AssertionError) as e:
            writeAndFlushStdErr("Powering down failed with error:\n{E}\n"
                                .format(E=e))

    # After doing a data run with the cavity's RF on we also do a run with the
    # electric heaters turned up by a known amount. This is used to reduce the
    # error in our calculated RF heat load due to the JT valve not being at
    # exactly the correct position to keep the liquid level steady over time,
    # which would show up as an extra term in the heat load.
    def launchHeaterRun(self, desPos):
        # type: (Cavity, float) -> datetime

        print("**** REMINDER: refills aren't automated - please contact the"
              " cryo group ****")

        self.walkHeater(CAV_HEATER_RUN_LOAD)

        if (self.liquidLevelDS - MIN_DS_LL) < TARGET_LL_DIFF:
            print("Please ask the cryo group to refill to {LL} on the"
                  " downstream sensor".format(LL=MAX_DS_LL))

            self.waitForCryo(desPos)

        startTime = datetime.now()

        print("\nStart time: {START}".format(START=startTime))

        writeAndWait(
            "\nWaiting for the LL to drop {DIFF}% or below {MIN}%...".format(
                MIN=MIN_DS_LL, DIFF=TARGET_LL_DIFF))

        startingLevel = self.liquidLevelDS
        avgLevel = startingLevel

        while ((startingLevel - avgLevel) < TARGET_LL_DIFF
               and (avgLevel > MIN_DS_LL)):
            writeAndWait(".", 15)
            avgLevel = self.liquidLevelDS

        endTime = datetime.now()

        print("\nEnd Time: {END}".format(END=endTime))
        duration = (endTime - startTime).total_seconds() / 3600
        print("Duration in hours: {DUR}".format(DUR=duration))

        self.walkHeater(-CAV_HEATER_RUN_LOAD)

        return endTime

    def runQ0Meas(self, desiredGradient, calibSession=None, valveParams=None):
        # type: (Cavity, float, CalibDataSession, ValveParams) -> (Q0DataSession, ValveParams)
        try:
            if not valveParams:
                valveParams = self.getRefValveParams()

            self.waitForCryo(valveParams.refValvePos)
            self.waitForTotalHeatDes(valveParams)

            self.checkAcqControl()
            self.setPowerStateSSA(True)
            self.characterize()

            # Setting the RF low and ramping up is time consuming so we skip it
            # during testing
            if not TEST_MODE:
                caputPV(self.genAcclPV("SEL_ASET"), "15")

            # Start with pulsed mode
            self.setModeRF("4")

            self.setStateRF(True)
            self.pushGoButton()

            self.checkAndSetOnTime()
            self.checkAndSetDrive()

            self.phaseCavity()

            if not TEST_MODE:
                self.lowerAmplitude()

            # go to CW
            self.setModeRF("2")

            self.walkToGradient(desiredGradient)

            startTime = self.holdGradient(desiredGradient).replace(microsecond=0)

            self.powerDown()

            endTime = self.launchHeaterRun(valveParams.refValvePos).replace(microsecond=0)

            timeParams = TimeParams(startTime, endTime, MYSAMPLER_TIME_INTERVAL)

            session = self.addDataSession(timeParams, valveParams,
                                          refGradVal=desiredGradient,
                                          calibSession=calibSession)

            with open(self.idxFile, 'a') as f:
                csvWriter = writer(f)
                csvWriter.writerow(
                    [self.cavNum, desiredGradient, valveParams.refValvePos,
                     startTime.strftime("%m/%d/%y %H:%M:%S"),
                     endTime.strftime("%m/%d/%y %H:%M:%S"),
                     valveParams.refHeatLoadDes,
                     valveParams.refHeatLoadAct,
                     MYSAMPLER_TIME_INTERVAL])

            print("\nStart Time: {START}".format(START=startTime))
            print("End Time: {END}".format(END=endTime))

            duration = (endTime - startTime).total_seconds() / 3600
            print("Duration in hours: {DUR}".format(DUR=duration))

            return session, valveParams

        except(CalledProcessError, IndexError, OSError, ValueError,
               AssertionError, KeyboardInterrupt) as e:
            writeAndFlushStdErr(
                "Procedure failed with error:\n{E}\n".format(E=e))
            self.powerDown()


class DataSession(object):
    __metaclass__ = ABCMeta

    def __init__(self, container, timeParams, valveParams):
        # type: (Container, TimeParams, ValveParams) -> None

        self.container = container
        self.dataRuns = []  # type: List[DataRun]
        self.heaterRunIdxs = []
        self.rfRunIdxs = []

        self._pvBuffMap = None

        self._dataFileName = None
        self._numPoints = None

        self.timeParams = timeParams
        self.valveParams = valveParams

        self.unixTimeBuff = []
        self.timeBuff = []
        self.valvePosBuff = []
        self.dsLevelBuff = []
        self.gradBuff = []
        self.dsPressBuff = []
        self.elecHeatDesBuff = []
        self.elecHeatActBuff = []

        # The plot of the raw downstream liquid level data
        self.liquidVsTimeAxis = None

    def __hash__(self):
        return self.hash(self.timeParams, self.container.cryModNumSLAC,
                         self.container.cryModNumJLAB)

    def __str__(self):
        return ("{START} to {END} ({RATE}s sample interval)"
                .format(START=self.timeParams.startTime,
                        END=self.timeParams.endTime,
                        RATE=self.timeParams.timeInterval))

    @property
    @abstractmethod
    def calibSlope(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def heatAdjustment(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def fileName(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def fileNameFormatter(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def adjustedRunSlopes(self):
        raise NotImplementedError

    # Takes the data in an session's buffers and slices it into data "runs"
    # based on cavity heater settings
    @abstractmethod
    def populateRuns(self):
        raise NotImplementedError

    # Iterates over this session's data runs, plots them, and fits trend lines
    # to them
    @abstractmethod
    def processRuns(self):
        raise NotImplementedError

    # Takes three related arrays, plots them, and fits some trend lines
    @abstractmethod
    def plotAndFitData(self):
        raise NotImplementedError

    @abstractmethod
    def getTotalHeatDelta(self, startIdx, currIdx):
        raise NotImplementedError

    @abstractmethod
    def printSessionReport(self):
        raise NotImplementedError

    @property
    def pvBuffMap(self):
        # type: () -> Dict[str, List[Union[datetime, float]]]
        if not self._pvBuffMap:
            raise NotImplementedError

        return self._pvBuffMap

    @property
    def runElecHeatLoads(self):
        # type: () -> List[float]
        return [self.dataRuns[runIdx].elecHeatLoadAct for runIdx
                in self.heaterRunIdxs]

    @property
    def numPoints(self):
        # type: () -> int
        if not self._numPoints:
            self._numPoints = int((self.timeParams.endTime
                                   - self.timeParams.startTime).total_seconds()
                                  / self.timeParams.timeInterval)
        return self._numPoints

    ############################################################################
    # A hash is effectively a unique numerical identifier. The purpose of a
    # hash function is to generate an ID for an object. In this case, we
    # consider data sessions to be identical if they have the same start & end
    # timeStamps, mySampler time interval, and cryomodule numbers. This function
    # takes all of those parameters and XORs (the ^ symbol) them.
    #
    # What is an XOR? It's an operator that takes two bit strings and goes
    # through them, bit by bit, returning True (1) only if one bit is 0 and the
    # other is 1
    #
    # EX) consider the following two bit strings a, b, and c = a^b:
    #       a: 101010010010 (2706 in base 10)
    #       b: 100010101011 (2219)
    #       ---------------
    #       c: 001000111001 (569)
    #
    # What we're doing here is taking each input data object's built-in hash
    # function (which returns an int) and XORing those ints together. It's not
    # QUITE unique, but XOR is the accepted way to hash in Python because
    # collisions are extremely rare (especially considering how many inputs we
    # have)
    #
    # As to WHY we're doing this, it's to have an easy way to compare
    # two data sessions so that we can avoid creating (and storing) duplicate
    # data sessions in the Container
    ############################################################################
    @staticmethod
    def hash(timeParams, slacNum, jlabNum, calibSession=None, refGradVal=None):
        # type: (TimeParams, int, int, CalibDataSession, float) -> int
        return (hash(timeParams.startTime) ^ hash(timeParams.endTime)
                ^ hash(timeParams.timeInterval) ^ hash(slacNum)
                ^ hash(jlabNum) ^ hash(calibSession) ^ hash(refGradVal))

    # Generates a CSV data file (with the raw data from this data session) if
    # one doesn't already exist
    def generateCSV(self):
        # type: () -> Optional[str]

        if isfile(self.fileName):
            return self.fileName

        (header, heaterActCols, heaterDesCols, colsToDelete,
         csvReader) = getDataAndHeaterCols(self.timeParams.startTime,
                                           self.numPoints,
                                           self.container.heaterDesPVs,
                                           self.container.heaterActPVs,
                                           self.container.getPVs(),
                                           self.timeParams.timeInterval)

        if not csvReader:
            return None

        else:

            # We're collapsing the readback for each cavity's desired and actual
            # electric heat load into two sum columns (instead of 16 individual
            # columns)
            compatibleMkdirs(self.fileName)
            with open(self.fileName, 'w+') as f:
                csvWriter = writer(f, delimiter=',')
                csvWriter.writerow(header)

                for row in csvReader:
                    (heatLoadSetpoint,
                     heatLoadAct) = collapseHeaterVals(row, heaterDesCols,
                                                       heaterActCols)

                    for index in colsToDelete:
                        del row[index]

                    row.append(str(heatLoadSetpoint))
                    row.append(str(heatLoadAct))
                    csvWriter.writerow(row)

            return self.fileName

    def processData(self):
        # type: () -> None

        self.parseDataFromCSV()
        self.dsLevelBuff = medfilt(self.dsLevelBuff, NUM_LL_POINTS_TO_AVG)
        self.populateRuns()

        if not self.dataRuns:
            print("{name} has no runs to process and plot."
                  .format(name=self.container.name))
            return

        self.adjustForSettle()
        self.processRuns()
        self.plotAndFitData()

    # parses CSV data to populate the given session's data buffers
    def parseDataFromCSV(self):
        # type: () -> None
        def linkBuffToColumn(column, dataBuff, headerRow):
            try:
                columnDict[column] = {"idx": headerRow.index(column),
                                      "buffer": dataBuff}
            except ValueError:
                writeAndFlushStdErr("Column " + column + " not found in CSV\n")

        columnDict = {}

        # noinspection PyTypeChecker
        with open(self.fileName) as csvFile:

            csvReader = reader(csvFile)
            header = compatibleNext(csvReader)

            # Figures out the CSV column that has that PV's data and maps it
            for pv, dataBuffer in self.pvBuffMap.items():
                linkBuffToColumn(pv, dataBuffer, header)

            linkBuffToColumn("Electric Heat Load Setpoint",
                             self.elecHeatDesBuff, header)

            linkBuffToColumn("Electric Heat Load Readback",
                             self.elecHeatActBuff, header)

            try:
                # Data fetched from the JLab archiver has the timestamp column
                # labeled "Date"
                timeIdx = header.index("Date")
                datetimeFormatStr = "%Y-%m-%d-%H:%M:%S"

            except ValueError:
                # Data exported from MyaPlot has the timestamp column labeled
                # "time"
                timeIdx = header.index("time")
                datetimeFormatStr = "%Y-%m-%d %H:%M:%S"

            timeZero = datetime.utcfromtimestamp(0)

            for row in csvReader:
                dt = datetime.strptime(row[timeIdx], datetimeFormatStr)

                self.timeBuff.append(dt)

                # We use Unix time to make the math easier during data
                # processing
                self.unixTimeBuff.append((dt - timeZero).total_seconds())

                # Actually parsing the CSV data into the buffers
                for col, idxBuffDict in columnDict.items():
                    try:
                        idxBuffDict["buffer"].append(
                            float(row[idxBuffDict["idx"]]))
                    except ValueError:
                        writeAndFlushStdErr("Could not fill buffer: " + str(col)
                                            + "\n")
                        idxBuffDict["buffer"].append(None)

    ############################################################################
    # adjustForSettle cuts off data that's corrupted because the heat load on
    # the 2 K helium bath is changing. (When the cavity heater settings or the
    # RF gradients change, it takes time for that change to become visible to
    # the helium because there are intermediate structures with heat capacity.)
    ############################################################################
    def adjustForSettle(self):
        # type: () -> None

        for i, run in enumerate(self.dataRuns):

            startIdx = run.startIdx

            totalHeatDelta = self.getTotalHeatDelta(startIdx, i)

            # Calculate the number of data points to be chopped off the
            # beginning of the data run based on the expected change in the
            # cryomodule heat load. The scale factor is derived from the
            # assumption that a 1 W change in the heat load leads to about 25
            # useless seconds (and that this scales linearly with the change in
            # heat load, which isn't really true). We already wait 30 s after
            # walking the heaters, so that's subtracted out
            # noinspection PyTypeChecker
            cutoff = int(totalHeatDelta * 25) - 30
            cutoff = cutoff if cutoff >= 0 else 0

            run.diagnostics["Cutoff"] = cutoff

            idx = self.dataRuns[i].startIdx
            startTime = self.unixTimeBuff[idx]
            duration = 0

            while duration < cutoff:
                idx += 1
                duration = self.unixTimeBuff[idx] - startTime

            self.dataRuns[i].startIdx = idx

    def _isEndOfCalibRun(self, idx, elecHeatLoadDes):
        # type: (int, float) -> bool
        # Find inflection points for the desired heater setting
        prevElecHeatLoadDes = (self.elecHeatDesBuff[idx - 1]
                               if idx > 0 else elecHeatLoadDes)

        heaterChanged = (elecHeatLoadDes != prevElecHeatLoadDes)
        liqLevelTooLow = (self.dsLevelBuff[idx] < MIN_DS_LL)
        valveOutsideTol = (abs(self.valvePosBuff[idx]
                               - self.valveParams.refValvePos)
                           > VALVE_POS_TOL)
        isLastElement = (idx == len(self.elecHeatDesBuff) - 1)

        heatersOutsideTol = (abs(elecHeatLoadDes - self.elecHeatActBuff[idx])
                             >= HEATER_TOL)

        # A "break" condition defining the end of a run if the desired heater
        # value changed, or if the upstream liquid level dipped below the
        # minimum, or if the valve position moved outside the tolerance, or if
        # we reached the end (which is a kinda jank way of "flushing" the last
        # run)
        return (heaterChanged or liqLevelTooLow or valveOutsideTol
                or isLastElement or heatersOutsideTol)

    def _checkAndFlushRun(self, isEndOfRun, idx, runStartIdx):
        # type: (bool, int, int) -> int
        if isEndOfRun:
            runDuration = (self.unixTimeBuff[idx]
                           - self.unixTimeBuff[runStartIdx])

            if runDuration >= MIN_RUN_DURATION:
                self._addRun(runStartIdx, idx - 1)

            return idx

        return runStartIdx

    @abstractmethod
    def _addRun(self, startIdx, endIdx):
        # type: (int, int) -> None
        raise NotImplementedError


class CalibDataSession(DataSession):

    def __init__(self, container, timeParams, valveParams):
        # type: (Cryomodule, TimeParams, ValveParams) -> None

        super(CalibDataSession, self).__init__(container, timeParams,
                                               valveParams)

        self.dataRuns = []  # type: List[HeaterDataRun]

        # Overloading these to give the IDE type hints
        self.container = container

        self._pvBuffMap = {self.container.valvePV: self.valvePosBuff,
                           self.container.dsLevelPV: self.dsLevelBuff}

        self._calibSlope = None

        # If we choose the JT valve position correctly, the calibration curve
        # should intersect the origin (0 heat load should translate to 0
        # dLL/dt). The heat adjustment will be equal to the negative x
        # intercept.
        self._heatAdjustment = None

        # the dLL/dt vs heat load plot with trend line (back-calculated points
        # for cavity Q0 sessions are added later)
        self.heaterCalibAxis = None

        self.generateCSV()
        self.processData()

    @property
    def calibSlope(self):
        # type: () -> float
        return self._calibSlope

    @property
    def heatAdjustment(self):
        # type: () -> float
        return self._heatAdjustment

    # returns a list of electric heat loads corrected with self.heatAdjustment
    @property
    def runElecHeatLoadsAdjusted(self):
        # type: () -> List[float]
        # noinspection PyUnresolvedReferences
        return [self.dataRuns[runIdx].elecHeatLoadActAdjusted for runIdx
                in self.heaterRunIdxs]

    @property
    def fileNameFormatter(self):
        # type: () -> str
        # e.g. calib_CM12_2019-02-25--11-25_18672.csv
        return "data/calib/cm{CM}/calib_{cryoMod}{suff}"

    @property
    def adjustedRunSlopes(self):
        # type: () -> List[float]
        return [self.dataRuns[runIdx].slope for runIdx in self.heaterRunIdxs]

    @property
    def fileName(self):
        # type: () -> str
        if not self._dataFileName:
            # Define a file name for the CSV we're saving. There are calibration
            # files and q0 measurement files. Both include a time stamp in the
            # format year-month-day--hour-minute. They also indicate the number
            # of data points.
            suffixStr = "{start}{nPoints}.csv"
            suffix = suffixStr.format(
                start=self.timeParams.startTime.strftime("_%Y-%m-%d--%H-%M_"),
                nPoints=self.numPoints)
            cryoModStr = "CM{cryMod}".format(
                cryMod=self.container.cryModNumSLAC)

            self._dataFileName = self.fileNameFormatter.format(
                cryoMod=cryoModStr, suff=suffix,
                CM=self.container.cryModNumSLAC)

        return self._dataFileName

    def populateRuns(self):
        # type: () -> None
        runStartIdx = 0

        for idx, elecHeatLoad in enumerate(self.elecHeatDesBuff):
            runStartIdx = self._checkAndFlushRun(
                self._isEndOfCalibRun(idx, elecHeatLoad), idx, runStartIdx)

    def _addRun(self, startIdx, endIdx):
        # type: (int, int) -> None
        runIdx = len(self.dataRuns)
        runNum = runIdx + 1

        self.dataRuns.append(HeaterDataRun(startIdx, endIdx, self, runNum))
        self.heaterRunIdxs.append(runIdx)

    # noinspection PyTupleAssignmentBalance
    def processRuns(self):
        # type: () -> None
        for run in self.dataRuns:
            run.process()

        # We're dealing with a cryomodule here so we need to calculate the
        # fit for the heater calibration curve.
        self._calibSlope, yIntercept = polyfit(self.runElecHeatLoads,
                                               self.adjustedRunSlopes, 1)

        xIntercept = -yIntercept / self._calibSlope

        self._heatAdjustment = -xIntercept

    def plotAndFitData(self):
        # type: () -> None
        # TODO improve plots with human-readable time axis

        plt.rcParams.update({'legend.fontsize': 'small'})

        suffix = " ({name} Heater Calibration)".format(name=self.container.name)

        self.liquidVsTimeAxis = genAxis("Liquid Level vs. Time" + suffix,
                                        "Unix Time (s)",
                                        "Downstream Liquid Level (%)")

        for run in self.dataRuns:
            # First we plot the actual run data
            self.liquidVsTimeAxis.plot(run.timeStamps, run.data,
                                       label=run.label)

            # Then we plot the linear fit to the run data
            self.liquidVsTimeAxis.plot(run.timeStamps, [run.slope * x
                                                        + run.intercept
                                                        for x
                                                        in run.timeStamps])

        self.liquidVsTimeAxis.legend(loc='best')
        self.heaterCalibAxis = genAxis("Liquid Level Rate of Change vs."
                                       " Heat Load", "Heat Load (W)",
                                       "dLL/dt (%/s)")

        self.heaterCalibAxis.plot(self.runElecHeatLoadsAdjusted,
                                  self.adjustedRunSlopes,
                                  marker="o", linestyle="None",
                                  label="Heater Calibration Data")

        slopeStr = '{:.2e}'.format(Decimal(self._calibSlope))
        labelStr = "Calibration Fit:  {slope} %/(s*W)".format(slope=slopeStr)

        self.heaterCalibAxis.plot(self.runElecHeatLoadsAdjusted,
                                  [self._calibSlope * x
                                   for x in self.runElecHeatLoadsAdjusted],
                                  label=labelStr)

        self.heaterCalibAxis.legend(loc='best')

    def getTotalHeatDelta(self, startIdx, currIdx):
        # type: (int, int) -> float
        if currIdx == 0:
            return self.elecHeatDesBuff[startIdx] - self.valveParams.refHeatLoadDes

        else:

            prevStartIdx = self.dataRuns[currIdx - 1].startIdx

            elecHeatDelta = (self.elecHeatDesBuff[startIdx]
                             - self.elecHeatDesBuff[prevStartIdx])

            return abs(elecHeatDelta)

    def printSessionReport(self):
        # type: () -> None

        print("\n-------------------------------------")
        print("---------------- {CM} ----------------"
              .format(CM=self.container.name))
        print("-------------------------------------\n")

        for run in self.dataRuns:
            run.printRunReport()

        print("Calibration curve intercept adjust = {ADJUST} W\n"
              .format(ADJUST=round(self.heatAdjustment, 4)))


class Q0DataSession(DataSession):

    def __init__(self, container, timeParams, valveParams, refGradVal,
                 calibSession):
        # type: (Cavity, TimeParams, ValveParams, float, CalibDataSession) -> None

        super(Q0DataSession, self).__init__(container, timeParams, valveParams)

        # Overloading these to give the IDE type hints
        self.container = container

        self._pvBuffMap = {self.container.parent.valvePV: self.valvePosBuff,
                           self.container.parent.dsLevelPV: self.dsLevelBuff,
                           self.container.gradPV: self.gradBuff,
                           self.container.parent.dsPressurePV: self.dsPressBuff}

        self.refGradVal = refGradVal
        self.calibSession = calibSession

        self.generateCSV()
        self.processData()

    def __hash__(self):
        return self.hash(self.timeParams, self.container.cryModNumSLAC,
                         self.container.cryModNumJLAB, self.calibSession,
                         self.refGradVal)

    @property
    def calibSlope(self):
        # type: () -> float
        return self.calibSession.calibSlope

    @property
    def heatAdjustment(self):
        # type: () -> float
        return self.calibSession.heatAdjustment

    @property
    def fileNameFormatter(self):
        # type: () -> str
        return "data/q0meas/cm{CM}/q0meas_{cryoMod}_cav{cavityNum}{suff}"

    # For Q0 data sessions we use the heater run(s) to calculate the heat
    # adjustment we should apply to the calculated RF heat load before
    # turning that into a Q0 value
    @property
    def avgHeatAdjustment(self):
        # type: () -> float
        adjustments = []

        for runIdx in self.heaterRunIdxs:
            # noinspection PyUnresolvedReferences
            runAdjustment = self.dataRuns[runIdx].heatAdjustment
            if runAdjustment:
                adjustments.append(runAdjustment)

        return mean(adjustments) if adjustments else 0

    # y = (m * x) + b, where y is the dLL/dt, x is the adjusted RF heat load,
    # and b is the y intercept for the calibration curve (which we normalized
    # to be 0). This is used when overlaying the RF run slopes on the
    # calibration curve.
    # noinspection PyUnresolvedReferences
    @property
    def adjustedRunSlopes(self):
        # type: () -> List[float]
        m = self.calibSession.calibSlope
        return [(m * self.dataRuns[runIdx].rfHeatLoad) for runIdx
                in self.rfRunIdxs]

    # noinspection PyUnresolvedReferences
    @property
    def adjustedRunHeatLoadsRF(self):
        # type: () -> List[float]
        return [self.dataRuns[runIdx].rfHeatLoad for runIdx
                in self.rfRunIdxs]

    @property
    def fileName(self):
        # type: () -> str
        if not self._dataFileName:
            suffixStr = "{start}{nPoints}.csv"
            suffix = suffixStr.format(
                start=self.timeParams.startTime.strftime("_%Y-%m-%d--%H-%M_"),
                nPoints=self.numPoints)
            cryoModStr = "CM{cryMod}".format(
                cryMod=self.container.cryModNumSLAC)

            self._dataFileName = self.fileNameFormatter.format(
                cryoMod=cryoModStr, suff=suffix,
                cavityNum=self.container.cavNum,
                CM=self.container.cryModNumSLAC)

        return self._dataFileName

    def _addRun(self, startIdx, endIdx):
        # type: (int, int) -> None
        runIdx = len(self.dataRuns)
        runNum = runIdx + 1

        isHeaterRun = (self.elecHeatDesBuff[startIdx] - self.valveParams.refHeatLoadDes) != 0
        if isHeaterRun:
            self.dataRuns.append(HeaterDataRun(startIdx, endIdx, self, runNum))
            self.heaterRunIdxs.append(runIdx)
        else:
            # noinspection PyTypeChecker
            self.dataRuns.append(RFDataRun(startIdx, endIdx, self, runNum))
            self.rfRunIdxs.append(runIdx)

    def populateRuns(self):
        # type: () -> None

        runStartIdx = 0

        for idx, elecHeatLoad in enumerate(self.elecHeatDesBuff):

            try:
                gradChanged = (abs(self.gradBuff[idx] - self.gradBuff[idx - 1])
                               > GRAD_TOL) if idx != 0 else False
            except TypeError:
                gradChanged = False

            isEndOfQ0Run = (self._isEndOfCalibRun(idx, elecHeatLoad)
                            or gradChanged)

            runStartIdx = self._checkAndFlushRun(isEndOfQ0Run, idx, runStartIdx)

    def processRuns(self):
        # type: () -> None
        for run in self.dataRuns:
            run.process()

    def plotAndFitData(self):
        # type: () -> None
        # TODO improve plots with human-readable time axis

        plt.rcParams.update({'legend.fontsize': 'small'})

        suffix = " ({name})".format(name=self.container.name)

        self.liquidVsTimeAxis = genAxis("Liquid Level vs. Time" + suffix,
                                        "Unix Time (s)",
                                        "Downstream Liquid Level (%)")

        for run in self.dataRuns:
            # First we plot the actual run data
            self.liquidVsTimeAxis.plot(run.timeStamps, run.data,
                                       label=run.label)

            # Then we plot the linear fit to the run data
            self.liquidVsTimeAxis.plot(run.timeStamps, [(run.slope * x)
                                                        + run.intercept
                                                        for x
                                                        in run.timeStamps])

        self.liquidVsTimeAxis.legend(loc='best')

    def getTotalHeatDelta(self, startIdx, currIdx):
        # type: (int, int) -> float
        if currIdx == 0:
            totalHeatDelta = (self.elecHeatDesBuff[startIdx] - self.valveParams.refHeatLoadDes)
            totalHeatDelta += self.approxHeatFromGrad(self.gradBuff[startIdx])
            return totalHeatDelta

        else:

            prevStartIdx = self.dataRuns[currIdx - 1].startIdx

            elecHeatDelta = (self.elecHeatDesBuff[startIdx]
                             - self.elecHeatDesBuff[prevStartIdx])

            currGrad = self.gradBuff[startIdx]
            currGradHeatLoad = self.approxHeatFromGrad(currGrad)

            prevGrad = self.gradBuff[prevStartIdx]
            prevGradHeatLoad = self.approxHeatFromGrad(prevGrad)

            gradHeatDelta = currGradHeatLoad - prevGradHeatLoad
            return abs(elecHeatDelta + gradHeatDelta)

    # Approximates the expected heat load on a cavity from its RF gradient. A
    # cavity with the design Q of 2.7E10 should produce about 9.6 W of heat with
    # a gradient of 16 MV/m. The heat scales quadratically with the gradient. We
    # don't know the correct Q yet when we call this function so we assume the
    # design values.
    @staticmethod
    def approxHeatFromGrad(grad):
        # type: (float) -> float
        # Gradients < 0 are non-physical so assume no heat load in that case.
        # The gradient values we're working with are readbacks from cavity
        # gradient PVs so it's possible that they could go negative.
        return ((grad / 16) ** 2) * 9.6 if grad > 0 else 0

    def updateOutput(self):
        self.printSessionReport()
        self.updateCalibCurve()

    def printSessionReport(self):
        # type: () -> None

        print("\n--------------------------------------")
        print("------------ {CM} {CAV} ------------"
              .format(CM=self.container.parent.name, CAV=self.container.name))
        print("--------------------------------------\n")

        for run in self.dataRuns:
            run.printRunReport()

    def updateCalibCurve(self):
        # type: () -> None

        calibSession = self.calibSession
        calibCurveAxis = calibSession.heaterCalibAxis

        calibCurveAxis.plot(self.adjustedRunHeatLoadsRF,
                            self.adjustedRunSlopes,
                            marker="o", linestyle="None",
                            label="Projected Data for " + self.container.name)

        calibCurveAxis.legend(loc='best', shadow=True, numpoints=1)

        # The rest of this mess is pretty much just extending the fit line to
        # include outliers
        minCavHeatLoad = min(self.adjustedRunHeatLoadsRF)
        minCalibHeatLoad = min(calibSession.runElecHeatLoadsAdjusted)

        if minCavHeatLoad < minCalibHeatLoad:
            yRange = linspace(minCavHeatLoad, minCalibHeatLoad)
            calibCurveAxis.plot(yRange, [calibSession.calibSlope * i
                                         for i in yRange])

        maxCavHeatLoad = max(self.adjustedRunHeatLoadsRF)
        maxCalibHeatLoad = max(calibSession.runElecHeatLoadsAdjusted)

        if maxCavHeatLoad > maxCalibHeatLoad:
            yRange = linspace(maxCalibHeatLoad, maxCavHeatLoad)
            calibCurveAxis.plot(yRange, [calibSession.calibSlope * i
                                         for i in yRange])


class DataRun(object):
    __metaclass__ = ABCMeta

    def __init__(self, runStartIdx, runEndIdx, dataSession, num):
        # type: (int, int, DataSession, int) -> None

        self.dataSession = dataSession
        self.num = num

        # startIdx and endIdx define the beginning and the end of this data run
        # within the cryomodule or cavity's data buffers
        self.startIdx = runStartIdx
        self.endIdx = runEndIdx

        self.elecHeatLoadDes = (dataSession.elecHeatDesBuff[runStartIdx]
                                - dataSession.valveParams.refHeatLoadDes)

        runElecHeatActBuff = self.dataSession.elecHeatActBuff[self.startIdx:
                                                              self.endIdx]

        self.heatActDelta = (mean(runElecHeatActBuff)
                             - self.dataSession.valveParams.refHeatLoadAct)

        # All data runs have liquid level information which gets fitted with a
        # line (giving us dLL/dt). The slope and intercept parametrize the line.
        self.slope = None
        self.intercept = None

        # A dictionary with some diagnostic information that only gets printed
        # if we're in test mode
        self.diagnostics = {}

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def adjustedTotalHeatLoad(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def label(self):
        raise NotImplementedError

    @abstractmethod
    def printRunReport(self):
        raise NotImplementedError

    # noinspection PyTypeChecker
    @property
    def elecHeatLoadAct(self):
        # type: () -> float
        return self.heatActDelta

    @property
    def data(self):
        # type: () -> List[float]
        return self.dataSession.dsLevelBuff[self.startIdx:self.endIdx]

    @property
    def timeStamps(self):
        # type: () -> List[float]
        return self.dataSession.unixTimeBuff[self.startIdx:self.endIdx]

    def genElecLabel(self):
        # type: () -> str
        labelStr = "{slope} %/s @ {heatLoad} W Electric Load"
        return labelStr.format(slope='%.2E' % Decimal(self.slope),
                               heatLoad=round(self.elecHeatLoadAct, 2))

    def process(self):
        # type: () -> None
        # noinspection PyTupleAssignmentBalance
        self.slope, self.intercept, r_val, p_val, std_err = linregress(
            self.timeStamps, self.data)

        self.diagnostics["R^2"] = r_val ** 2

        startTime = self.dataSession.unixTimeBuff[self.startIdx]
        endTime = self.dataSession.unixTimeBuff[self.endIdx]
        self.diagnostics["Duration"] = ((endTime - startTime) / 60.0)

    def printDiagnostics(self):
        # type: () -> None

        print("            Cutoff: {CUT}"
              .format(CUT=self.diagnostics["Cutoff"]))

        print("          Duration: {DUR}"
              .format(DUR=round(self.diagnostics["Duration"], 4)))

        # Print R^2 for the run's fit line to diagnose whether or not it was
        # long enough
        print("               R^2: {R2}\n"
              .format(R2=round(self.diagnostics["R^2"], 4)))


class HeaterDataRun(DataRun):

    def __init__(self, runStartIdx, runEndIdx, dataSession, num):
        # type: (int, int, DataSession, int) -> None

        super(HeaterDataRun, self).__init__(runStartIdx, runEndIdx, dataSession,
                                            num)
        self.dataSession = dataSession

    @property
    def name(self):
        # type: () -> str
        return "Run {NUM} ({TYPE})".format(NUM=self.num, TYPE="heater")

    @property
    def adjustedTotalHeatLoad(self):
        # type: () -> float
        return self.elecHeatLoadAct

    # Heat error due to the position of the JT valve
    @property
    def heatAdjustment(self):
        # type: () -> float
        calcHeatLoad = (self.slope / self.dataSession.calibSlope)
        return self.elecHeatLoadAct - calcHeatLoad

    @property
    def elecHeatLoadActAdjusted(self):
        # type: () -> float
        return self.heatActDelta + self.dataSession.heatAdjustment

    @property
    def label(self):
        # type: () -> str
        return self.genElecLabel()

    def printRunReport(self):
        # type: () -> None

        print("   ------- Run {NUM} (Heater) -------\n".format(NUM=self.num))

        reportStr = "     Electric heat load: {ELEC} W\n"
        report = reportStr.format(ELEC=round(self.elecHeatLoadAct, 2))

        # print(report.format(Q0Val=None))
        print(report)

        # if TEST_MODE:
        #    self.printDiagnostics()
        self.printDiagnostics()


class RFDataRun(DataRun):

    def __init__(self, runStartIdx, runEndIdx, dataSession, num):
        # type: (int, int, Q0DataSession, int) -> None

        super(RFDataRun, self).__init__(runStartIdx, runEndIdx, dataSession,
                                        num)

        # Stores the average RF gradient for this run
        self.grad = None

        self._calculatedQ0 = None
        self.dataSession = dataSession

    @property
    def name(self):
        # type: () -> str
        return "Run {NUM} ({TYPE})".format(NUM=self.num, TYPE="RF")

    # Each Q0 measurement run has a total heat load value. If it is an RF run
    # we calculate the heat load by projecting the run's dLL/dt on the
    # cryomodule's heater calibration curve. If it is a heater run we just
    # return the electric heat load.
    @property
    def adjustedTotalHeatLoad(self):
        # type: () -> float
        return ((self.slope / self.dataSession.calibSlope)
                + self.dataSession.avgHeatAdjustment)

    # The RF heat load is equal to the total heat load minus the electric
    # heat load.
    @property
    def rfHeatLoad(self):
        # type: () -> float
        return self.adjustedTotalHeatLoad - self.elecHeatLoadAct

    @property
    def q0(self):
        # type: () -> float

        if not self._calculatedQ0:
            q0s = []
            numInvalidGrads = 0
            calcFile = "calculations/cm{NUM}/cav{CAV}.csv".format(NUM=self.dataSession.container.cryModNumSLAC,
                                                                  CAV=self.dataSession.container.cavNum)

            compatibleMkdirs(calcFile)
            with open(calcFile, "w+") as f:
                csvWriter = writer(f, delimiter=',')
                csvWriter.writerow(["Gradient", "RF Heat Load", "Pressure",
                                    "Q0"])

                for idx in range(self.startIdx, self.endIdx):
                    archiveGrad = self.dataSession.gradBuff[idx]

                    if archiveGrad:
                        q0s.append(self.calcQ0(archiveGrad, self.rfHeatLoad,
                                               self.dataSession.dsPressBuff[idx]))
                        csvWriter.writerow([archiveGrad, self.rfHeatLoad,
                                            self.dataSession.dsPressBuff[idx], q0s[-1]])

                    # Sometimes the archiver messes up and records 0 for some
                    # reason. We use the reference desired value as an approximation
                    else:
                        numInvalidGrads += 1
                        q0s.append(self.calcQ0(self.dataSession.refGradVal,
                                               self.rfHeatLoad,
                                               self.dataSession.dsPressBuff[idx]))
                        csvWriter.writerow([self.dataSession.refGradVal, self.rfHeatLoad,
                                            self.dataSession.dsPressBuff[idx], q0s[-1]])

                if numInvalidGrads:
                    writeAndFlushStdErr("\nGradient buffer had {NUM} invalid points"
                                        " (used reference gradient value instead) "
                                        "- Consider refetching the data from the "
                                        "archiver\n"
                                        .format(NUM=numInvalidGrads))

            self._calculatedQ0 = float(mean(q0s))

        return self._calculatedQ0

    @property
    def label(self):
        # type: () -> str

        labelStr = "{slope} %/s @ {grad} MV/m\nCalculated Q0: {Q0}"
        q0Str = '{:.2e}'.format(Decimal(self.q0))

        return labelStr.format(slope='%.2E' % Decimal(self.slope),
                               grad=self.dataSession.refGradVal, Q0=q0Str)

    def printRunReport(self):
        # type: () -> None

        print("    --------- Run {NUM} (RF) ---------\n".format(NUM=self.num))

        reportStr = ("      Avg Pressure: {PRES} Torr\n"
                     "       RF Gradient: {GRAD} MV/m\n"
                     "      RF heat load: {RFHEAT} W\n"
                     "   Heat Adjustment: {ADJUST} W\n"
                     "     Calculated Q0: {Q0Val}\n")

        avgPress = mean(self.dataSession.dsPressBuff[self.startIdx:self.endIdx])

        gradVals = self.dataSession.gradBuff[self.startIdx:self.endIdx]
        rmsGrad = sqrt(sum(g ** 2 for g in gradVals)
                       / (self.endIdx - self.startIdx))

        heatAdjust = self.dataSession.avgHeatAdjustment

        Q0 = '{:.2e}'.format(Decimal(self.q0))

        # noinspection PyTypeChecker
        report = reportStr.format(PRES=round(avgPress, 2),
                                  GRAD=round(rmsGrad, 2),
                                  RFHEAT=round(self.rfHeatLoad, 2),
                                  ADJUST=round(heatAdjust, 2),
                                  Q0Val=Q0)

        print(report)

        # if TEST_MODE:
        #    self.printDiagnostics()
        self.printDiagnostics()

    # The calculated Q0 value for this run. Magical formula from Mike Drury
    # (drury@jlab.org) to calculate Q0 from the measured heat load on a cavity,
    # the RF gradient used during the test, and the pressure of the incoming
    # 2 K helium.
    @staticmethod
    def calcQ0(grad, rfHeatLoad, avgPressure):
        # type: (float, float, float) -> float
        # The initial Q0 calculation doesn't account for the temperature
        # variation of the 2 K helium
        uncorrectedQ0 = ((grad * 1000000) ** 2) / (939.3 * rfHeatLoad)

        # We can correct Q0 for the helium temperature!
        tempFromPress = (avgPressure * 0.0125) + 1.705

        C1 = 271
        C2 = 0.0000726
        C3 = 0.00000214
        C4 = grad - 0.7
        C5 = 0.000000043
        C6 = -17.02
        C7 = C2 - (C3 * C4) + (C5 * (C4 ** 2))

        return (C1 / ((C7 / 2) * exp(C6 / 2) + C1 / uncorrectedQ0
                      - (C7 / tempFromPress) * exp(C6 / tempFromPress)))


def main():
    cryomodule = Cryomodule(cryModNumSLAC=12, cryModNumJLAB=2)
    for idx, cav in cryomodule.cavities.items():
        print(cav.gradPV)


if __name__ == '__main__':
    main()
