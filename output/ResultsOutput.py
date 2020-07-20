# -*- coding: utf-8 -*-
# @File         : ResultsOutput.py
# @Author       : Zhendong Zhang
# @Organization : Huazhong University of Science and Technology
# @Email        : zzd_zzd@hust.edu.cn
# @Time         : 2020/7/3 15:14
import forecast
from util.const.GlobalConstants import GlobalConstants
from util.io.InputOuputUtils import NumpyInputOutputUtils, PandasInputOutputUtils
import pandas as pd
import numpy as np
import os

from util.tool.ToolUtils import ToolNormalUtils


class ResultsOutput(object):
    @staticmethod
    def outputOrgData(orgData, savePath, sheetName):
        NumpyInputOutputUtils.saveNumpyToExcel(orgData, savePath, sheetName)

    @staticmethod
    def outputSmallData(smallData, savePath, sheetName):
        NumpyInputOutputUtils.saveNumpyToExcel(smallData, savePath, sheetName)

    @staticmethod
    def outputDatasetAfterFeatureOperation(datasetX, datasetY, featureNamesAfterOperation, savePath, sheetName):
        datasetX = pd.DataFrame(datasetX, columns=featureNamesAfterOperation)
        datasetY = pd.DataFrame(datasetY, columns=["L"])
        dataset = pd.concat([datasetY, datasetX], axis=1)
        PandasInputOutputUtils.savePandasToExcel(dataset, savePath, sheetName)

    @staticmethod
    def outputMetricInFeatureSelection(metricValues, featureNames, metricNames, savePath, sheetName):
        if metricValues is not None:
            metricNames = [metricName.name for metricName in metricNames]
            pandasData = pd.DataFrame(np.array(metricValues).T, columns=featureNames, index=metricNames)
            PandasInputOutputUtils.savePandasToExcel(pandasData, savePath, sheetName)

    @staticmethod
    def outputFeatureOptimizationProcess(featureOptimizationProcess, featureNamesAfterOperation, savePath, sheetName):
        if featureOptimizationProcess is not None:
            objTrace = featureOptimizationProcess["objTrace"]
            varTrace = featureOptimizationProcess["varTrace"]
            obj = pd.DataFrame(objTrace, columns=["平均/当前 适应度", "最优/当前 适应度"])
            var = pd.DataFrame(varTrace, columns=featureNamesAfterOperation)
            dataset = pd.concat([obj, var], axis=1)
            PandasInputOutputUtils.savePandasToExcel(dataset, savePath, sheetName)

    @staticmethod
    def outputHyperParametersOptimizationProcess(hyperParametersOptimizationProcess, savePath):
        if hyperParametersOptimizationProcess is not None:
            best = hyperParametersOptimizationProcess["best"]
            trials = hyperParametersOptimizationProcess["trials"]
            optimizationProcess = trials.results
            times = len(optimizationProcess)
            # 优化过程中损失值
            losses = np.zeros(shape=(times, 1))
            bestLosses = np.zeros(shape=(times, 1))
            for i in range(times):
                optimizationResult = optimizationProcess[i]
                losses[i] = optimizationResult['loss']
                if i == 0:
                    bestLosses[i] = losses[i]
                else:
                    if bestLosses[i - 1] <= losses[i]:
                        bestLosses[i] = bestLosses[i - 1]
                    else:
                        bestLosses[i] = losses[i]
            lossProcess = np.append(losses, bestLosses, axis=1)  # axis=1 横向扩展
            lossProcess = pd.DataFrame(lossProcess, columns=['losses', 'bestLosses'])
            PandasInputOutputUtils.savePandasToExcel(lossProcess, savePath, "lossProcess")
            # 超参数组合与损失值对应
            colNames = best.keys()
            allData = dict()
            for colName in colNames:
                data = []
                for i in range(times):
                    optimizationResult = optimizationProcess[i]
                    data.append(optimizationResult['params'][colName])
                allData[colName] = data
            allDataFrame = pd.DataFrame(allData)
            lossesDataFrame = pd.DataFrame(losses[:, 0], columns=['losses'])
            allDataFrame = pd.concat([allDataFrame, lossesDataFrame], axis=1)
            PandasInputOutputUtils.savePandasToExcel(allDataFrame, savePath, "hyperParametersProcess")

    @staticmethod
    def outputFinalHyperParameters(forecastModel, savePath):
        optimizedHyperParameters = forecastModel.getOptimizedHyperParameters()
        fixedHyperParameters = forecastModel.getFixedHyperParameters()
        optimalParamsFrame = pd.DataFrame(optimizedHyperParameters, index=[0])
        PandasInputOutputUtils.savePandasToExcel(optimalParamsFrame, savePath, "optimizedHyperParameters")
        # 采用的固定的超参数
        fixedParamsFrame = pd.DataFrame(fixedHyperParameters, index=[0])
        PandasInputOutputUtils.savePandasToExcel(fixedParamsFrame, savePath, "fixedHyperParameters")

    @staticmethod
    def pickleFinalForecastModel(loss, forecastModel):
        # 是否更新参数的标致
        isUpdate = False
        finalForecastModelPath, bestLossPath = GlobalConstants.getFinalForecastModelAndBestLossPath()
        if os.path.exists(finalForecastModelPath):
            bestLoss = ToolNormalUtils.loadData(bestLossPath)
            if loss < bestLoss:
                ToolNormalUtils.pickleData(bestLossPath, loss)
                ToolNormalUtils.pickleData(finalForecastModelPath, forecastModel)
                isUpdate = True
        else:
            ToolNormalUtils.pickleData(bestLossPath, loss)
            ToolNormalUtils.pickleData(finalForecastModelPath, forecastModel)
            isUpdate = True
        return isUpdate

    @staticmethod
    def pickleFinalHyperParametersOptimizationProcess(finalHyperParametersOptimizationProcess):
        finalHyperParametersOptimizationProcessPath = GlobalConstants.getFinalHyperParametersOptimizationProcessPath()
        ToolNormalUtils.pickleData(finalHyperParametersOptimizationProcessPath, finalHyperParametersOptimizationProcess)

    @staticmethod
    def outputDeterministicForecastingResults(predictions, observations, savePath):
        observationsDataFrame = pd.DataFrame(observations, columns=["真实值"])
        predictionsDataFrame = pd.DataFrame(predictions, columns=["预报值"])
        allDataFrame = pd.concat([observationsDataFrame, predictionsDataFrame], axis=1)
        PandasInputOutputUtils.savePandasToExcel(allDataFrame, savePath, "确定性预报结果")

    @staticmethod
    def outputIntervalForecastingResults(dataset, probabilisticForecastingResults, savePath):
        if probabilisticForecastingResults is not None:
            observations = dataset.validationY.flatten()
            observations = dataset.reverseLabel(observations)
            observations = observations.flatten()
            intervalData = []
            rowNames = []
            for alpha in [0.8, 0.85, 0.9, 0.95]:
                lowerAlpha = (1 - alpha) / 2
                upperAlpha = 1 - lowerAlpha

                lower = forecast.Forecast.ForecastModelBase.getPredictionsByQuantile(probabilisticForecastingResults, lowerAlpha)
                upper = forecast.Forecast.ForecastModelBase.getPredictionsByQuantile(probabilisticForecastingResults, upperAlpha)
                lower = lower.flatten()
                upper = upper.flatten()
                intervalData.append(lower)
                intervalData.append(upper)
                rowNames.append(str(alpha)+"下限")
                rowNames.append(str(alpha) + "上限")
            intervalData.append(observations)
            rowNames.append("真实值")
            intervalData = np.array(intervalData).T
            intervalDataFrame = pd.DataFrame(intervalData, columns=rowNames)
            PandasInputOutputUtils.savePandasToExcel(intervalDataFrame, savePath, "区间预测结果")

    @staticmethod
    def outputProbabilisticForecastingResults(probabilisticForecastingResults, savePath):
        if probabilisticForecastingResults is not None:
            pdfs = probabilisticForecastingResults["pdfs"]
            cdfs = probabilisticForecastingResults["cdfs"]
            sampleNum = pdfs.shape[0]
            rowNames = []
            for i in range(sampleNum):
                pdf = pdfs[i]
                if i == 0:
                    pdfsNumpy = np.array([pdf["x"]])
                    pdfsNumpy = np.r_[pdfsNumpy, [pdf["f"]]]
                else:
                    pdfsNumpy = np.r_[pdfsNumpy, [pdf["x"]]]
                    pdfsNumpy = np.r_[pdfsNumpy, [pdf["f"]]]
                rowNames.append("x")
                rowNames.append("f")
            pdfsDataFrame = pd.DataFrame(pdfsNumpy, index=rowNames)
            PandasInputOutputUtils.savePandasToExcel(pdfsDataFrame, savePath, "pdfs")
            rowNames = []
            for i in range(sampleNum):
                cdf = cdfs[i]
                if i == 0:
                    cdfsNumpy = np.array([cdf["x"]])
                    cdfsNumpy = np.r_[cdfsNumpy, [cdf["F"]]]
                else:
                    cdfsNumpy = np.r_[cdfsNumpy, [cdf["x"]]]
                    cdfsNumpy = np.r_[cdfsNumpy, [cdf["F"]]]
                rowNames.append("x")
                rowNames.append("F")
            cdfsDataFrame = pd.DataFrame(cdfsNumpy, index=rowNames)
            PandasInputOutputUtils.savePandasToExcel(cdfsDataFrame, savePath, "cdfs")

    @staticmethod
    def outputForecastingMetrics(metrics, metricNames, savePath, sheetName):
        metricDataFrame = pd.DataFrame(np.array(metrics).reshape(1, -1), columns=metricNames)
        PandasInputOutputUtils.savePandasToExcel(metricDataFrame, savePath, sheetName)
