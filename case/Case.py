# -*- coding: utf-8 -*-
# @File         : Case.py
# @Author       : Zhendong Zhang
# @Organization : Huazhong University of Science and Technology
# @Email        : zzd_zzd@hust.edu.cn
# @Time         : 2020/7/2 19:47
from dataset.Dataset import SmallDataFactory, Dataset
from forecast.Feature import FeatureGeneration, FeatureSelection, FeatureCombination, FeatureOptimization
from forecast.Forecast import ForecastTemplate, ForecastModelBase
from scipy import stats
import numpy as np
import random
import time
from tensorflow import set_random_seed

from metric.Metric import ForecastPerformanceMetricFactory, ForecastPerformanceMetricEnum
from output.Draw import Draw
from output.ResultsOutput import ResultsOutput
from util.const.GlobalConstants import GlobalConstants
from util.io.InputOuputUtils import NumpyInputOutputUtils
from util.tool.ToolUtils import ToolNormalUtils


class CaseProcess(object):

    def __init__(self, caseParams):
        timeFlag = str(int(time.time()))
        timeFlag = "19941111"
        rootPath = GlobalConstants.getRootPath()

        # 获取基础设置
        baseSetting = caseParams["baseSetting"]
        caseFlag = baseSetting["caseFlag"]
        caseDescription = baseSetting["caseDescription"]
        isFixedRandomSeed = baseSetting["isFixedRandomSeed"]
        if isFixedRandomSeed:
            fixedRandomSeed = baseSetting["fixedRandomSeed"]
            # 整个项目用到随机数的地方太多了，并且引用的包也多，难以将所有随机过程都设置随机种子
            # 有些引用的包内不一定暴露了设置随机种子的接口，所以有些时候即使采用了固定随机种子也可能会失效
            # 下面已经尽可能把用到的随机种子都设置了，大概率能复现了，但不敢保证100%
            random.seed(fixedRandomSeed)
            np.random.seed(fixedRandomSeed)
            set_random_seed(fixedRandomSeed)
        runTimes = baseSetting["runTimes"]

        caseFlagPath = rootPath + "resources/" + caseFlag + "/"
        timeFlagPath = caseFlagPath + timeFlag + "/"

        # 获取数据集设置
        datasetSettings = caseParams["datasetSettings"]

        # 获取预报模型设置
        forecastModelSettings = caseParams["forecastModelSettings"]

        # 案例运行流程
        datasetNum = len(datasetSettings)
        forecastModelNum = len(forecastModelSettings)
        for i in range(runTimes):
            # 第i次运行
            runTimePath = timeFlagPath + "run" + str(i + 1) + "/"
            for j in range(datasetNum):
                # 第j个数据集
                datasetSetting = datasetSettings[j]["datasetSetting"]

                datasetFlag = datasetSetting["datasetFlag"]
                datasetDescription = datasetSetting["datasetDescription"]
                datasetPath = rootPath + datasetSetting["datasetPath"]
                datasetSubPath = datasetSetting["datasetSubPath"]
                smallDataParams = datasetSetting["smallDataParams"]
                featureParams = datasetSetting["featureParams"]

                datasetFlagPath = runTimePath + datasetFlag + "/"
                datasetSuffix = "_" + timeFlag + "_" + str(i + 1) + "_" + str(j + 1)
                # 读取数据
                orgData = NumpyInputOutputUtils.readExcelDataToNumpy(datasetPath, datasetSubPath)
                orgDataPath = datasetFlagPath + "common/" + "orgData" + datasetSuffix + ".xlsx"
                isOutputOrgData = datasetSetting["isOutputOrgData"]
                if isOutputOrgData:
                    ResultsOutput.outputOrgData(orgData, orgDataPath, "orgData")
                # 大数据集到小数据
                smallData = SmallDataFactory.getSmallData(orgData, smallDataParams)
                smallDataPath = datasetFlagPath + "common/" + "smallData" + datasetSuffix + ".xlsx"
                isOutputSmallData = datasetSetting["isOutputSmallData"]
                if isOutputSmallData:
                    ResultsOutput.outputSmallData(smallData, smallDataPath, "smallData")
                # 小数据数据生成特征和标签
                featureGenerationParams = featureParams["featureGenerationParams"]
                featureGeneration = FeatureGeneration(smallData, featureGenerationParams)
                self.datasetAfterGenerationX, self.datasetAfterGenerationY, self.featureNamesAfterGeneration = featureGeneration.getDatasetAfterGeneration()
                isOutputDatasetAfterFeatureGeneration = featureParams["isOutputDatasetAfterFeatureGeneration"]
                datasetAfterFeatureGenerationPath = datasetFlagPath + "common/" + "datasetAfterFeatureGeneration" + datasetSuffix + ".xlsx"
                if isOutputDatasetAfterFeatureGeneration:
                    ResultsOutput.outputDatasetAfterFeatureOperation(self.datasetAfterGenerationX,
                                                                     self.datasetAfterGenerationY,
                                                                     self.featureNamesAfterGeneration,
                                                                     datasetAfterFeatureGenerationPath,
                                                                     "datasetAfterFeatureGeneration")
                # 将生成的特征进行初选
                featureSelectionParams = featureParams["featureSelectionParams"]
                featureSelection = FeatureSelection(self.datasetAfterGenerationX, self.datasetAfterGenerationY,
                                                    self.featureNamesAfterGeneration, featureSelectionParams)
                self.datasetAfterSelectionX, self.datasetAfterSelectionY, self.featureNamesAfterSelection, self.metricValues = featureSelection.getDatasetAfterSelection()
                isOutputDatasetAfterFeatureSelection = featureParams["isOutputDatasetAfterFeatureSelection"]
                datasetAfterFeatureSelectionPath = datasetFlagPath + "common/" + "datasetAfterFeatureSelection" + datasetSuffix + ".xlsx"
                if isOutputDatasetAfterFeatureSelection:
                    ResultsOutput.outputDatasetAfterFeatureOperation(self.datasetAfterSelectionX,
                                                                     self.datasetAfterSelectionY,
                                                                     self.featureNamesAfterSelection,
                                                                     datasetAfterFeatureSelectionPath,
                                                                     "datasetAfterFeatureSelection")
                isOutputMetricInFeatureSelection = featureParams["isOutputMetricInFeatureSelection"]
                metricInFeatureSelectionPath = datasetFlagPath + "common/" + "metricInFeatureSelection" + datasetSuffix + ".xlsx"
                if isOutputMetricInFeatureSelection:
                    ResultsOutput.outputMetricInFeatureSelection(self.metricValues, self.featureNamesAfterGeneration,
                                                                 featureSelectionParams["operateEnums"],
                                                                 metricInFeatureSelectionPath,
                                                                 "metricInFeatureSelection")
                # 手动进行特征组合
                featureCombinationParams = featureParams["featureCombinationParams"]
                featureCombination = FeatureCombination(self.datasetAfterSelectionX, self.datasetAfterSelectionY,
                                                        self.featureNamesAfterSelection, featureCombinationParams)
                self.datasetAfterCombinationX, self.datasetAfterCombinationY, self.featureNamesAfterCombination = featureCombination.getDatasetAfterCombination()
                isOutputDatasetAfterFeatureCombination = featureParams["isOutputDatasetAfterFeatureCombination"]
                datasetAfterFeatureCombinationPath = datasetFlagPath + "common/" + "datasetAfterFeatureCombination" + datasetSuffix + ".xlsx"
                if isOutputDatasetAfterFeatureCombination:
                    ResultsOutput.outputDatasetAfterFeatureOperation(self.datasetAfterCombinationX,
                                                                     self.datasetAfterCombinationY,
                                                                     self.featureNamesAfterCombination,
                                                                     datasetAfterFeatureCombinationPath,
                                                                     "datasetAfterFeatureCombination")

                for k in range(forecastModelNum):
                    # 第k个预报模型

                    # 预报模型设置
                    forecastModelSetting = forecastModelSettings[k]["forecastModelSetting"]
                    forecastModelParams = forecastModelSetting["forecastModelParams"]
                    forecastModelEnum = forecastModelParams["forecastModelEnum"]
                    forecastModelSuffix = datasetSuffix + "_" + forecastModelEnum.name
                    finalForecastModelPath = datasetFlagPath + forecastModelEnum.name + "/" + "finalForecastModel" + forecastModelSuffix + ".txt"
                    bestLossPath = datasetFlagPath + forecastModelEnum.name + "/" + "bestLoss" + forecastModelSuffix + ".txt"
                    finalHyperParametersOptimizationProcessPath = datasetFlagPath + forecastModelEnum.name + "/" + "finalHyperParametersOptimizationProcess" + forecastModelSuffix + ".txt"
                    GlobalConstants.setFinalForecastModelAndBestLossPath(finalForecastModelPath, bestLossPath)
                    GlobalConstants.setFinalHyperParametersOptimizationProcessPath(
                        finalHyperParametersOptimizationProcessPath)
                    ToolNormalUtils.deleteFile(bestLossPath)
                    ToolNormalUtils.deleteFile(finalForecastModelPath)
                    ToolNormalUtils.deleteFile(finalHyperParametersOptimizationProcessPath)
                    # 数据集设置
                    datasetParams = datasetSetting["datasetParams"]
                    # 特征优化: 特征优化和超参数优化必须同步进行，因此这步骤里完成了超参数优化和最终模型的优化
                    featureOptimizationParams = featureParams["featureOptimizationParams"]
                    featureOptimization = FeatureOptimization(self.datasetAfterCombinationX,
                                                              self.datasetAfterCombinationY,
                                                              self.featureNamesAfterCombination,
                                                              featureOptimizationParams, datasetParams,
                                                              forecastModelParams)
                    self.datasetAfterOptimizationX, self.datasetAfterOptimizationY, self.featureNamesAfterOptimization, self.featureOptimizationProcess = featureOptimization.getDatasetAfterOptimization()
                    isOutputDatasetAfterFeatureOptimization = featureParams["isOutputDatasetAfterFeatureOptimization"]
                    datasetAfterFeatureOptimizationPath = datasetFlagPath + forecastModelEnum.name + "/" + "datasetAfterFeatureOptimization" + forecastModelSuffix + ".xlsx"
                    if isOutputDatasetAfterFeatureOptimization:
                        ResultsOutput.outputDatasetAfterFeatureOperation(self.datasetAfterOptimizationX,
                                                                         self.datasetAfterOptimizationY,
                                                                         self.featureNamesAfterOptimization,
                                                                         datasetAfterFeatureOptimizationPath,
                                                                         "datasetAfterFeatureOptimization")
                    isOutputFeatureOptimizationProcess = featureParams["isOutputFeatureOptimizationProcess"]
                    featureOptimizationProcessPath = datasetFlagPath + forecastModelEnum.name + "/" + "featureOptimizationProcess" + forecastModelSuffix + ".xlsx"
                    if isOutputFeatureOptimizationProcess:
                        ResultsOutput.outputFeatureOptimizationProcess(self.featureOptimizationProcess,
                                                                       self.featureNamesAfterCombination,
                                                                       featureOptimizationProcessPath,
                                                                       "featureOptimizationProcess")
                    # 特征优化和超参数优化需要同时进行，并且要及时保存双优化整个过程中的最优结果
                    # 因为即使获取了最优特征和最优超参数，但由于随机因素存在，不同时间启动训练预报模型导致预测结果与双优化过程中的最优结果无法完全相同
                    # 采用最优特征和最优超参数来再次训练和预报，只能说平均来讲比其他特征和参数大概率会优

                    # 构造数据集
                    self.finalDataset = Dataset(self.datasetAfterOptimizationX, self.datasetAfterOptimizationY,
                                                datasetParams)
                    isOutputFinalDataset = datasetParams["isOutputFinalDataset"]
                    finalDatasetPath = datasetFlagPath + forecastModelEnum.name + "/" + "finalDataset" + forecastModelSuffix + ".xlsx"
                    if isOutputFinalDataset:
                        ResultsOutput.outputDatasetAfterFeatureOperation(self.finalDataset.trainX2D,
                                                                         self.finalDataset.trainY2D,
                                                                         self.featureNamesAfterOptimization,
                                                                         finalDatasetPath,
                                                                         "trainSet")
                        ResultsOutput.outputDatasetAfterFeatureOperation(self.finalDataset.validationX2D,
                                                                         self.finalDataset.validationY2D,
                                                                         self.featureNamesAfterOptimization,
                                                                         finalDatasetPath,
                                                                         "validationSet")
                    # 完成 超参数优化-->模型训练  这步骤在特征优化中已经完成，取出来存和用就可以了
                    self.finalForecastModel = ToolNormalUtils.loadData(finalForecastModelPath)
                    self.hyperParametersOptimizationProcess = ToolNormalUtils.loadData(
                        finalHyperParametersOptimizationProcessPath)
                    isPerformOptimization = forecastModelParams["isPerformOptimization"]
                    isOutputHyperParametersOptimizationProcess = forecastModelParams[
                        "isOutputHyperParametersOptimizationProcess"]
                    hyperParametersOptimizationProcessPath = datasetFlagPath + forecastModelEnum.name + "/" + "hyperParametersOptimizationProcess" + forecastModelSuffix + ".xlsx"
                    if isPerformOptimization and isOutputHyperParametersOptimizationProcess:
                        ResultsOutput.outputHyperParametersOptimizationProcess(self.hyperParametersOptimizationProcess,
                                                                               hyperParametersOptimizationProcessPath)
                    isOutputFinalHyperParameters = forecastModelParams["isOutputFinalHyperParameters"]
                    finalHyperParametersPath = datasetFlagPath + forecastModelEnum.name + "/" + "finalHyperParameters" + forecastModelSuffix + ".xlsx"
                    if isOutputFinalHyperParameters:
                        ResultsOutput.outputFinalHyperParameters(self.finalForecastModel, finalHyperParametersPath)
                    # 完成最终的预报
                    # 获取整个优化过程中的最优模型来进行最终预报
                    predictions = self.finalForecastModel.predict(isFlatten=True)
                    observations = self.finalForecastModel.dataset.validationY.flatten()
                    predictionsReverse = self.finalDataset.reverseLabel(predictions)
                    observationsReverse = self.finalDataset.reverseLabel(observations)
                    isOutputDeterministicForecastingResults = forecastModelParams[
                        "isOutputDeterministicForecastingResults"]
                    deterministicForecastingResultsPath = datasetFlagPath + forecastModelEnum.name + "/" + "deterministicForecastingResults" + forecastModelSuffix + ".xlsx"
                    if isOutputDeterministicForecastingResults:
                        ResultsOutput.outputDeterministicForecastingResults(predictionsReverse, observationsReverse,
                                                                            deterministicForecastingResultsPath)
                    isOutputDeterministicForecastingMetrics = forecastModelParams[
                        "isOutputDeterministicForecastingMetrics"]
                    forecastMetricsPath = datasetFlagPath + forecastModelEnum.name + "/" + "forecastMetrics" + forecastModelSuffix + ".xlsx"
                    if isOutputDeterministicForecastingMetrics:
                        pointMetricEnums = [ForecastPerformanceMetricEnum.PointForecastMetricR2,
                                            ForecastPerformanceMetricEnum.PointForecastMetricRMSE,
                                            ForecastPerformanceMetricEnum.PointForecastMetricMSE,
                                            ForecastPerformanceMetricEnum.PointForecastMetricMAPE,
                                            ForecastPerformanceMetricEnum.PointForecastMetricMAE]
                        pointMetrics = []
                        pointMetricNames = []
                        for pointMetricEnum in pointMetricEnums:
                            metric = ForecastPerformanceMetricFactory.getPointForecastMetric(
                                self.finalForecastModel.dataset, pointMetricEnum)
                            pointMetrics.append(metric)
                            pointMetricNames.append(pointMetricEnum.name)
                        ResultsOutput.outputForecastingMetrics(pointMetrics, pointMetricNames, forecastMetricsPath,
                                                               "点预测指标")

                    # 概率预报
                    isPerformProbabilisticForecasting = forecastModelParams["isPerformProbabilisticForecasting"]
                    if isPerformProbabilisticForecasting:
                        probabilisticForecastModelParams = forecastModelParams["probabilisticForecastModelParams"]
                        probabilisticResults = self.finalForecastModel.getProbabilisticResults(
                            probabilisticForecastModelParams)
                        isOutputProbabilisticForecastingResults = forecastModelParams[
                            "isOutputProbabilisticForecastingResults"]
                        probabilisticForecastingResultsPath = datasetFlagPath + forecastModelEnum.name + "/" + "probabilisticForecastingResults" + forecastModelSuffix + ".xlsx"
                        intervalForecastingResultsPath = datasetFlagPath + forecastModelEnum.name + "/" + "intervalForecastingResults" + forecastModelSuffix + ".xlsx"
                        if isOutputProbabilisticForecastingResults:
                            ResultsOutput.outputProbabilisticForecastingResults(probabilisticResults,
                                                                                probabilisticForecastingResultsPath)
                            ResultsOutput.outputIntervalForecastingResults(self.finalDataset, probabilisticResults,
                                                                           intervalForecastingResultsPath)

                        isOutputProbabilisticForecastingMetrics = forecastModelParams[
                            "isOutputProbabilisticForecastingMetrics"]
                        if isOutputProbabilisticForecastingMetrics:
                            intervalMetricEnums = [ForecastPerformanceMetricEnum.IntervalForecastMetricCP,
                                                   ForecastPerformanceMetricEnum.IntervalForecastMetricMWP,
                                                   ForecastPerformanceMetricEnum.IntervalForecastMetricCM,
                                                   ForecastPerformanceMetricEnum.IntervalForecastMetricMC]
                            intervalMetrics = []
                            intervalMetricNames = []
                            for alpha in [0.8, 0.85, 0.9, 0.95]:
                                for intervalMetricEnum in intervalMetricEnums:
                                    metric = ForecastPerformanceMetricFactory.getIntervalForecastMetric(
                                        self.finalForecastModel.dataset, intervalMetricEnum, alpha)
                                    intervalMetrics.append(metric)
                                    intervalMetricNames.append(intervalMetricEnum.name + "_" + str(alpha))
                            ResultsOutput.outputForecastingMetrics(intervalMetrics, intervalMetricNames,
                                                                   forecastMetricsPath, "区间预测指标")

                            probabilisticMetricEnums = [ForecastPerformanceMetricEnum.ProbabilisticForecastMetricCRPS,
                                                        ForecastPerformanceMetricEnum.ProbabilisticForecastMetricPIT]
                            probabilisticMetrics = []
                            probabilisticMetricNames = []
                            for probabilisticMetricEnum in probabilisticMetricEnums:
                                metric = ForecastPerformanceMetricFactory.getProbabilisticForecastMetric(
                                    self.finalForecastModel.dataset, probabilisticMetricEnum)
                                probabilisticMetrics.append(metric)
                                probabilisticMetricNames.append(probabilisticMetricEnum.name)
                            ResultsOutput.outputForecastingMetrics(probabilisticMetrics, probabilisticMetricNames,
                                                                   forecastMetricsPath, "概率预测指标")

                    # 绘图
                    isShowPredictionPlots = forecastModelParams["isShowPredictionPlots"]
                    isSavePredictionPlots = forecastModelParams["isSavePredictionPlots"]
                    labelName = forecastModelParams["labelName"]
                    drawSuffix = "_" + datasetFlag + "_" + forecastModelEnum.name + "_" + timeFlag + "_" + str(
                        i + 1) + "_" + str(j + 1)
                    savePredictionPath = datasetFlagPath + "plots/" + "predictions" + drawSuffix + ".jpg"
                    if isPerformProbabilisticForecasting:
                        lower = ForecastModelBase.getPredictionsByQuantile(probabilisticResults, 0.05)
                        upper = ForecastModelBase.getPredictionsByQuantile(probabilisticResults, 0.95)
                        alpha = "90%"
                    else:
                        lower = None
                        upper = None
                        alpha = None
                    title = "predictions" + drawSuffix
                    Draw.drawPredictions(predictionsReverse, observationsReverse, lower, upper, alpha, "时段", labelName,
                                         title, isShowPredictionPlots, isSavePredictionPlots, savePredictionPath)
                    if isPerformProbabilisticForecasting:
                        isShowProbabilisticPlots = forecastModelParams["isShowProbabilisticPlots"]
                        isSaveProbabilisticPlots = forecastModelParams["isSaveProbabilisticPlots"]
                        isShowReliablePlots = forecastModelParams["isShowReliablePlots"]
                        isSaveReliablePlots = forecastModelParams["isSaveReliablePlots"]
                        pdfs = probabilisticResults["pdfs"]
                        cdfs = probabilisticResults["cdfs"]
                        sampleNum = len(pdfs)
                        periods = [0, round(0.5 * sampleNum), sampleNum - 1]
                        for period in periods:
                            pdf = pdfs[period]
                            cdf = cdfs[period]
                            observation = observationsReverse[period]
                            title = "PDF_" + "period_" + str(period) + drawSuffix
                            saveProbabilisticPath = datasetFlagPath + "plots/" + "PDF_" + "period_" + str(
                                period) + drawSuffix + ".jpg"
                            Draw.drawPDForCDF(pdf["x"], pdf["f"], observation, labelName, "概率密度", title,
                                              isShowProbabilisticPlots,
                                              isSaveProbabilisticPlots, saveProbabilisticPath)
                            title = "CDF_" + "period_" + str(period) + drawSuffix
                            saveProbabilisticPath = datasetFlagPath + "plots/" + "CDF_" + "period_" + str(
                                period) + drawSuffix + ".jpg"
                            Draw.drawPDForCDF(cdf["x"], cdf["F"], observation, labelName, "累计分布", title,
                                              isShowProbabilisticPlots,
                                              isSaveProbabilisticPlots, saveProbabilisticPath)
                        title = "PIT" + drawSuffix
                        saveReliablePath = datasetFlagPath + "plots/" + "PIT" + drawSuffix + ".jpg"
                        pits = ForecastPerformanceMetricFactory.getPIT(cdfs, observationsReverse)
                        Draw.drawPIT(pits, stats.uniform, title=title, isShow=isShowReliablePlots,
                                     isSave=isSaveReliablePlots, savePath=saveReliablePath)
                    # pre5 = finalForecastModel.getPredictionsByQuantile(probabilisticResults, 0.5)
                    # daaa = finalForecastModel.dataset
                    # daaa.validationP = pre5
                    # print("0.5Q:" + str(
                    #     ForecastPerformanceMetricFactory.getPointForecastMetric(daaa,
                    #                                                             ForecastPerformanceMetricEnum.PointForecastMetricR2))
                    #       )
                    # draw = Draw()
                    # draw.drawPredictions(pre5, observations, None, None, None, False, "period", "zd",
                    #                      "dataset",
                    #                      [1.0, 0.35], True, False, None)

            print("CaseProcess-->__init__()中断点1")


class CaseTemplate(object):
    pass
