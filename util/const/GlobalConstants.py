# -*- coding: utf-8 -*-
# @File         : GlobalConstants.py
# @Author       : Zhendong Zhang
# @Organization : Huazhong University of Science and Technology
# @Email        : zzd_zzd@hust.edu.cn
# @Time         : 2020/6/28 12:58 
import os

import forecast
from dataset.Dataset import SmallDataOperateEnum


class GlobalConstants(object):
    """
    global constants for the entire project
    """

    finalForecastModelPath = ""
    bestLossPath = ""
    finalHyperParametersOptimizationProcessPath = ""

    @staticmethod
    def getRootPath():
        """
        The absolute root path of the entire project. It helps me write relative path like 'src' in Jave.
        :return
            ROOT_PATH : The absolute root path of the entire project.
        """

        projectName = "WSHexperiment"
        curPath = os.path.abspath(os.path.dirname(__file__))
        ROOT_PATH = curPath[:curPath.find(projectName + "\\") + len(projectName + "\\")]
        return ROOT_PATH

    @staticmethod
    def setFinalForecastModelAndBestLossPath(finalForecastModelPath, bestLossPath):
        GlobalConstants.finalForecastModelPath = finalForecastModelPath
        GlobalConstants.bestLossPath = bestLossPath

    @staticmethod
    def getFinalForecastModelAndBestLossPath():
        return GlobalConstants.finalForecastModelPath, GlobalConstants.bestLossPath

    @staticmethod
    def setFinalHyperParametersOptimizationProcessPath(finalHyperParametersOptimizationProcessPath):
        GlobalConstants.finalHyperParametersOptimizationProcessPath = finalHyperParametersOptimizationProcessPath

    @staticmethod
    def getFinalHyperParametersOptimizationProcessPath():
        return GlobalConstants.finalHyperParametersOptimizationProcessPath


class GlobalEnumConstants(object):
    """
    整个项目所有的枚举
    """

    @staticmethod
    def getEnum(enumClassString, enumElementString):
        allEnumList = [
            SmallDataOperateEnum,
            forecast.Forecast.ForecastModelEnum,
            forecast.OptimizationObjective.OptimizationObjectiveEnum]
        for enum in allEnumList:
            if enum.__name__ == enumClassString:
                for element in enum.__dict__["_member_map_"].items():
                    if element[0] == enumElementString:
                        return element[1]
        raise ValueError("错误的枚举类型：" + enumClassString + "." + enumElementString)
