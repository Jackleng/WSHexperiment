# -*- coding: utf-8 -*-
# @File         : Feature.py
# @Author       : Zhendong Zhang
# @Organization : Huazhong University of Science and Technology
# @Email        : zzd_zzd@hust.edu.cn
# @Time         : 2020/6/30 11:49
import random
from enum import Enum
import geatpy as ea
from geatpy.Problem import Problem
import numpy as np

from dataset.Dataset import Dataset
from forecast.Forecast import ForecastTemplate
from forecast.OptimizationObjective import OptimizationObjectiveFactory
from metric.Metric import CorrelationAnalysisMetricFactory, CorrelationAnalysisMetricEnum
from output.ResultsOutput import ResultsOutput
from util.const.GlobalConstants import GlobalConstants
from util.math.MathUtils import MathTransformUtils, MathNormalUtils
from util.tool.ToolUtils import NumpyUtils, ToolNormalUtils


class FeatureTemplate(object):
    def __init__(self, orgData, featureParams, datasetParams, forecastModelParams):
        self.orgData = orgData
        self.featureParams = featureParams
        self.datasetParams = datasetParams
        self.forecastModelParams = forecastModelParams

        # 将原始数据生成特征和标签
        featureGenerationParams = featureParams["featureGenerationParams"]
        featureGeneration = FeatureGeneration(orgData, featureGenerationParams)
        self.datasetAfterGenerationX, self.datasetAfterGenerationY = featureGeneration.getDatasetAfterGeneration()
        # 将生成的特征进行初选
        featureSelectionParams = featureParams["featureSelectionParams"]
        featureSelection = FeatureSelection(self.datasetAfterGenerationX, self.datasetAfterGenerationY,
                                            featureSelectionParams)
        self.datasetAfterSelectionX, self.datasetAfterSelectionY = featureSelection.getDatasetAfterSelection()
        # 手动进行特征组合
        featureCombinationParams = featureParams["featureCombinationParams"]
        featureCombination = FeatureCombination(self.datasetAfterSelectionX, self.datasetAfterSelectionY,
                                                featureCombinationParams)
        self.datasetAfterCombinationX, self.datasetAfterCombinationY = featureCombination.getDatasetAfterCombination()
        # 特征优化
        featureOptimizationParams = featureParams["featureOptimizationParams"]
        featureOptimization = FeatureOptimization(self.datasetAfterCombinationX, self.datasetAfterCombinationY,
                                                  featureOptimizationParams, datasetParams, forecastModelParams)
        self.datasetAfterOptimizationX, self.datasetAfterOptimizationY = featureOptimization.getDatasetAfterOptimization()

    def getFinalDataset(self):
        return self.datasetAfterOptimizationX, self.datasetAfterOptimizationY


class FeatureGeneration(object):

    def __init__(self, orgData, featureGenerationParams):
        self.orgData = orgData
        self.featureGenerationParams = featureGenerationParams
        # 特征生成
        self.datasetAfterGenerationX, self.datasetAfterGenerationY, self.featureNamesAfterGeneration = self.generate(
            orgData, featureGenerationParams)

    @staticmethod
    def generate(orgData, featureGenerationParams):
        """
        根据输入进来的数据集生成特征和标签
        :param orgData: 原始数据集，numpy二维数组，从SmallDataFactory获取
        :param featureGenerationParams: 用于特征生成的参数，dict类型，用什么value向字典中补充什么key，谁用谁维护，构造器输入变量就这两个，不再变化
                    isAdditionalGenerate: 是否额外生成，boolean
                    featureIndexes: 特征列索引，numpy一维数组，0为首计数
                    operateEnums: 每个特征列的特征生成方法，FeatureGenerationOperateEnum构成的numpy二维数组[i, j]:第i个特征列，第j个操作
                    transformFuncs: 每个特征列的特征生成方法，与operateEnums对应，String构成的numpy二维数组,每个String是一个函数表达式
                    historyIndexes: 每个生成后的特征列再取多少历史前期数据，numpy二维数组，a[i][j]:第i个特征列，第j种历史前期，a[i][j]表示前a[i][j]个时段的值
                    labelIndex: 一个Integer数，标签索引，0为首计数
        :return:
                datasetAfterGenerationX: 特征生成后的数据集特征
                datasetAfterGenerationY: 特征生成后的数据集标签
        """
        featureIndexes = featureGenerationParams["featureIndexes"]
        isAdditionalGenerate = featureGenerationParams["isAdditionalGenerate"]
        labelIndex = featureGenerationParams["labelIndex"]
        featureNamesAfterGeneration = []
        if not isAdditionalGenerate:
            datasetAfterGenerationX = None
            count = 0
            for featureIndex in featureIndexes:
                # 按列取值是时orgData[:, featureIndex:featureIndex+1]得到的是二维数组形式；orgData[:, featureIndex]一维数组形式
                colSeries = orgData[:, featureIndex:featureIndex + 1]
                X = colSeries
                count = count + 1
                if datasetAfterGenerationX is None:
                    datasetAfterGenerationX = X
                else:
                    datasetAfterGenerationX = NumpyUtils.hStackByCutHead(datasetAfterGenerationX, X)
                featureNamesAfterGeneration.append("_")
        else:
            operateEnums = featureGenerationParams["operateEnums"]
            transformFuncs = featureGenerationParams["transformFuncs"]
            historyIndexes = featureGenerationParams["historyIndexes"]
            datasetAfterGenerationX = None
            count = 0
            for featureIndex in featureIndexes:
                # 按列取值是时orgData[:, featureIndex:featureIndex+1]得到的是二维数组形式；orgData[:, featureIndex]一维数组形式
                colSeries = orgData[:, featureIndex:featureIndex + 1]
                historyIndex = historyIndexes[count]
                X, newFeatureNames = FeatureGenerationFactory.generateFeature(colSeries, operateEnums[count],
                                                                              transformFuncs[count],
                                                                              historyIndex)
                newFeatureNames = ["f" + str(featureIndex) + "_" + newFeatureName for newFeatureName in newFeatureNames]
                count = count + 1
                if datasetAfterGenerationX is None:
                    datasetAfterGenerationX = X
                else:
                    datasetAfterGenerationX = NumpyUtils.hStackByCutHead(datasetAfterGenerationX, X)
                featureNamesAfterGeneration.extend(newFeatureNames)
        sampleNum = datasetAfterGenerationX.shape[0]
        labSeries = orgData[:, labelIndex:labelIndex + 1]
        datasetAfterGenerationY = labSeries[-sampleNum::, :]

        featureNamesAfterGeneration = ["F" + str(i + 1) + "_" + featureNamesAfterGeneration[i] for i in
                                       range(len(featureNamesAfterGeneration))]

        if (datasetAfterGenerationX is None) or (datasetAfterGenerationY is None):
            raise Exception("特征生成失败！")

        return datasetAfterGenerationX, datasetAfterGenerationY, featureNamesAfterGeneration

    def getDatasetAfterGeneration(self):
        return self.datasetAfterGenerationX, self.datasetAfterGenerationY, self.featureNamesAfterGeneration


class FeatureGenerationFactory(object):
    """
    特征生成工厂
    """

    @staticmethod
    def generateFeature(colSeries, operateEnums, transformFuncs, historyIndex):
        newFeatures = None
        newFeatureNames = []
        maxIndex = max(historyIndex)  # 历史前期索引最大值
        T = colSeries.shape[0]  # 行数
        for i in range(len(operateEnums)):
            operateEnum = operateEnums[i]
            transformFunc = transformFuncs[i]
            if operateEnum == FeatureGenerationOperateEnum.Linear:
                newColSeries = MathTransformUtils.linearTransformFunc(colSeries, transformFunc)
            elif operateEnum == FeatureGenerationOperateEnum.Power:
                newColSeries = MathTransformUtils.powerTransformFunc(colSeries, transformFunc)
            else:
                raise ValueError("FeatureGenerationFactory-->generateFeature()中operateEnum参数错误，只能是：\n" + ("\n".join(
                    ['%s:%s' % item for item in FeatureGenerationOperateEnum.__dict__["_member_map_"].items()])))

            for j in historyIndex:
                newFeature = newColSeries[:T - j, :]
                if newFeatures is None:
                    newFeatures = newFeature
                else:
                    newFeatures = NumpyUtils.hStackByCutHead(newFeatures, newFeature)
                newFeatureName = transformFunc + "_t" + str(j)
                newFeatureNames.append(newFeatureName)

        return newFeatures, newFeatureNames


class FeatureGenerationOperateEnum(Enum):
    """
    特征生成操作枚举
    """

    # 线性变换
    Linear = 1
    # 幂变换
    Power = 2

    pass


class FeatureSelection(object):

    def __init__(self, datasetX, datasetY, featureNamesAfterGeneration, featureSelectionParams):
        self.datasetX = datasetX
        self.datasetY = datasetY
        self.featureNamesAfterGeneration = featureNamesAfterGeneration
        self.featureSelectionParams = featureSelectionParams
        # 特征选择
        self.datasetAfterSelectionX, self.datasetAfterSelectionY, self.featureNamesAfterSelection, self.metricValues = self.select(
            datasetX, datasetY, featureNamesAfterGeneration, featureSelectionParams)

    @staticmethod
    def select(datasetX, datasetY, featureNamesAfterGeneration, featureSelectionParams):
        """
        对特征进行简单筛选，比如一些极其不相关的特征可以在进行特征组合前进行删除以缩小特征组合优化的搜索空间
        此处的原则：剔除掉非常弱相关的特征，对于较弱相关的特征偏向保留，因为这些较弱特征可能在特征组合中发挥神奇的效果
        如果这类较弱相关的特征真的对预测精度没有贡献，在特征组合之后也会被删除掉，所以不用担心这步冗余了特征
        :param datasetX: 数据集特征，从FeatureGeneration过来的
        :param datasetY: 数据集标签，从FeatureGeneration过来的
        :param featureNamesAfterGeneration: 特征名称，从FeatureGeneration过来的
        :param featureSelectionParams: 用于特征选择的参数，dict类型，用什么value向字典中补充什么key，谁用谁维护，构造器输入变量就这两个，不再变化
                    isPerformSelection: 是否进行特征筛选, Boolean
                    operateEnums: 用于特征选择的方法，FeatureSelectionOperateEnum的numpy一维数组
                    transformFuncs: 多种特征选择指标之间是求并还是求交，[and, or]的String numpy一维数组
                    thresholdValues: 每种特征选择方法的阈值, numpy一维数组
        :return:
                datasetAfterSelectionX: 特征选择后的数据集特征
                datasetAfterSelectionY: 特征选择后的数据集标签
        """

        isPerformSelection = featureSelectionParams["isPerformSelection"]
        featureNamesAfterSelection = []
        metricValues = []
        if isPerformSelection:
            operateEnums = featureSelectionParams["operateEnums"]
            transformFuncs = featureSelectionParams["transformFuncs"]
            thresholdValues = featureSelectionParams["thresholdValues"]

            datasetAfterSelectionX = None
            datasetAfterSelectionY = datasetY
            colNum = datasetX.shape[1]
            for i in range(colNum):
                colSeries = datasetX[:, i:i + 1]
                selected, metricValue = FeatureSelectionFactory.selectFeature(colSeries, datasetY, operateEnums,
                                                                              transformFuncs,
                                                                              thresholdValues)
                metricValues.append(metricValue)
                if selected:
                    if datasetAfterSelectionX is None:
                        datasetAfterSelectionX = colSeries
                    else:
                        datasetAfterSelectionX = NumpyUtils.hStackByCutTail(datasetAfterSelectionX, colSeries)
                    featureNamesAfterSelection.append(featureNamesAfterGeneration[i])
        else:
            datasetAfterSelectionX = datasetX
            datasetAfterSelectionY = datasetY
            featureNamesAfterSelection = featureNamesAfterGeneration

        if (datasetAfterSelectionX is None) or (datasetAfterSelectionY is None):
            raise Exception("特征选择条件太过苛刻，请检查参数！")
        return datasetAfterSelectionX, datasetAfterSelectionY, featureNamesAfterSelection, metricValues

    def getDatasetAfterSelection(self):
        return self.datasetAfterSelectionX, self.datasetAfterSelectionY, self.featureNamesAfterSelection, self.metricValues


class FeatureSelectionFactory(object):
    """
    特征选择工厂
    """

    @staticmethod
    def selectFeature(colSeries, labelSeries, operateEnums, transformFuncs, thresholdValues):
        selected = True
        metricValues = []
        for i in range(len(operateEnums)):
            operateEnum = operateEnums[i]
            transformFunc = transformFuncs[i]
            thresholdValue = thresholdValues[i]
            singleSelected = None
            if operateEnum == FeatureSelectionOperateEnum.PCC:
                metric = CorrelationAnalysisMetricFactory.getMetric(colSeries, labelSeries,
                                                                    CorrelationAnalysisMetricEnum.PCC)
                metricValues.append(metric)
                singleSelected = metric >= thresholdValue
            elif operateEnum == FeatureSelectionOperateEnum.MIC:
                metric = CorrelationAnalysisMetricFactory.getMetric(colSeries, labelSeries,
                                                                    CorrelationAnalysisMetricEnum.MIC)
                metricValues.append(metric)
                singleSelected = metric >= thresholdValue
            else:
                raise ValueError("FeatureGenerationFactory-->generateFeature()中operateEnum参数错误，只能是：\n" + ("\n".join(
                    ['%s:%s' % item for item in FeatureGenerationOperateEnum.__dict__["_member_map_"].items()])))

            if transformFunc == "and":
                selected = selected and singleSelected
            elif transformFunc == "or":
                selected = selected or singleSelected
            else:
                raise ValueError("FeatureGenerationFactory-->generateFeature()中transformFuncs参数错误，只能是：[and, or]\n")

        return selected, metricValues


class FeatureSelectionOperateEnum(Enum):
    """
    特征选择操作枚举
    """

    # 皮尔逊相关系数的绝对值
    PCC = 1
    # 最大信息系数
    MIC = 2

    pass


class FeatureCombination(object):
    def __init__(self, datasetX, datasetY, featureNamesAfterSelection, featureCombinationParams):
        self.datasetX = datasetX
        self.datasetY = datasetY
        self.featureNamesAfterSelection = featureNamesAfterSelection
        self.featureCombinationParams = featureCombinationParams
        # 特征组合
        self.datasetAfterCombinationX, self.datasetAfterCombinationY, self.featureNamesAfterCombination = self.combine(
            datasetX, datasetY, featureNamesAfterSelection, featureCombinationParams)

    @staticmethod
    def combine(datasetX, datasetY, featureNamesAfterSelection, featureCombinationParams):
        """
        特征组合
        在一个特征组合中，原备选特征有被选中和不被选中两种状态，即等价于0-1规划问题
        这个类只针对某一特定的0-1组合值featureFlags返回对应的数据集，实际上也是在做筛选特征的事情
        而至于哪种特征组合featureFlags最优，是在FeatureOptimization中完成
        :param datasetX: 数据集特征，从FeatureSelection过来的
        :param datasetY: 数据集标签，从从FeatureSelection过来的过来的
        :param featureCombinationParams: 用于特征组合的参数，dict类型，用什么value向字典中补充什么key，谁用谁维护，构造器输入变量就这两个，不再变化
                    isPerformCombination: 是否进行特征组合, Boolean
                    featureFlags: 特征组合的标志，[0, 1]的numpy一维数组
        :return:
                datasetAfterCombinationX: 特征选择后的数据集特征
                datasetAfterCombinationY: 特征选择后的数据集标签
        """
        featureNamesAfterCombination = []
        isPerformCombination = featureCombinationParams["isPerformCombination"]
        if isPerformCombination:
            featureFlags = featureCombinationParams["featureFlags"]
            datasetAfterCombinationX = None
            datasetAfterCombinationY = datasetY
            colNum = datasetX.shape[1]
            for i in range(colNum):
                colSeries = datasetX[:, i:i + 1]
                if int(featureFlags[i]) == 1:
                    if datasetAfterCombinationX is None:
                        datasetAfterCombinationX = colSeries
                    else:
                        datasetAfterCombinationX = NumpyUtils.hStackByCutTail(datasetAfterCombinationX, colSeries)
                    featureNamesAfterCombination.append(featureNamesAfterSelection[i])
        else:
            datasetAfterCombinationX = datasetX
            datasetAfterCombinationY = datasetY
            featureNamesAfterCombination = featureNamesAfterSelection

        if (datasetAfterCombinationX is None) or (datasetAfterCombinationY is None):
            raise Exception("特征组合中全部特征未被选取，请检查参数！")
        return datasetAfterCombinationX, datasetAfterCombinationY, featureNamesAfterCombination

    def getDatasetAfterCombination(self):
        return self.datasetAfterCombinationX, self.datasetAfterCombinationY, self.featureNamesAfterCombination


class FeatureOptimization(object):
    def __init__(self, datasetX, datasetY, featureNamesAfterCombination, featureOptimizationParams, datasetParams,
                 forecastModelParams):
        self.datasetX = datasetX
        self.datasetY = datasetY
        self.featureNamesAfterCombination = featureNamesAfterCombination
        self.featureOptimizationParams = featureOptimizationParams
        self.datasetParams = datasetParams
        self.forecastModelParams = forecastModelParams
        self.datasetAfterOptimizationX, self.datasetAfterOptimizationY, self.featureNamesAfterOptimization, self.featureOptimizationProcess = self.optimize(
            datasetX, datasetY, featureNamesAfterCombination, featureOptimizationParams, datasetParams,
            forecastModelParams)

    @staticmethod
    def optimize(datasetX, datasetY, featureNamesAfterCombination, featureOptimizationParams, datasetParams,
                 forecastModelParams):
        featureNamesAfterOptimization = []
        featureOptimizationProcess = None
        isPerformOptimization = featureOptimizationParams["isPerformOptimization"]
        if isPerformOptimization:
            dim = len(featureNamesAfterCombination)
            problem = FeatureOptimizationProblem(datasetX, datasetY, featureNamesAfterCombination, datasetParams,
                                                 forecastModelParams)
            encoding = 'RI'
            popNum = 10
            field = ea.crtfld(encoding, problem.varTypes, problem.ranges, problem.borders)
            pop = ea.Population(encoding, field, popNum)

            if dim < 7:
                # 维度不高时采用穷举
                Phen = []
                for i in range(1, pow(2, dim)):
                    numStr = MathNormalUtils.toBinaryWithFixedLength(i, dim)
                    numArray = list(map(int, numStr))
                    Phen.append(numArray)
                # 受维数灾影响 随机优化5个特征组合
                resultlist = random.sample(range(len(Phen)), 3)
                Phen = np.array(Phen)
                # Phen = Phen[resultlist, :]
                pop.Phen = Phen
                problem.aimFunc(pop)
                objTrace, varTrace = pop.ObjV, pop.Phen
                objTrace = NumpyUtils.hStackByCutHead(objTrace, objTrace)
            else:

                algorithm = ea.soea_SEGA_templet(problem, pop)
                # 算法最大进化代数
                algorithm.MAXGEN = 10
                # 0表示不绘图；1表示绘图；2表示动态绘图
                algorithm.drawing = 1

                [pop, objTrace, varTrace] = algorithm.run()
            featureOptimizationProcess = {
                "objTrace": objTrace,
                "varTrace": varTrace
            }
            bestGen = np.argmin(problem.maxormins * objTrace[:, 1])
            bestVar = varTrace[bestGen, :]
            bestVar = [int(x) for x in bestVar]
            bestVar = np.array(bestVar)

            featureCombinationParams = {
                "isPerformCombination": True,
                "featureFlags": bestVar
            }
            featureCombination = FeatureCombination(datasetX, datasetY, featureNamesAfterCombination,
                                                    featureCombinationParams)
            datasetAfterOptimizationX, datasetAfterOptimizationY, featureNamesAfterOptimization = featureCombination.getDatasetAfterCombination()
        else:
            datasetAfterOptimizationX = datasetX
            datasetAfterOptimizationY = datasetY
            featureNamesAfterOptimization = featureNamesAfterCombination

            dataset = Dataset(datasetAfterOptimizationX, datasetAfterOptimizationY, datasetParams)
            forecastTemplate = ForecastTemplate(dataset, forecastModelParams)
            loss = forecastTemplate.getObjective()
        if (datasetAfterOptimizationX is None) or (datasetAfterOptimizationY is None):
            raise Exception("特征优化中全部特征未被选取，请检查参数！")
        return datasetAfterOptimizationX, datasetAfterOptimizationY, featureNamesAfterOptimization, featureOptimizationProcess

    def getDatasetAfterOptimization(self):
        return self.datasetAfterOptimizationX, self.datasetAfterOptimizationY, self.featureNamesAfterOptimization, self.featureOptimizationProcess


class FeatureOptimizationProblem(Problem):
    def __init__(self, datasetX, datasetY, featureNamesAfterCombination, datasetParams, forecastModelParams):
        self.datasetX = datasetX
        self.datasetY = datasetY
        self.featureNamesAfterCombination = featureNamesAfterCombination
        self.datasetParams = datasetParams
        self.forecastModelParams = forecastModelParams
        self.optimizationObjectiveEnum = forecastModelParams["optimizationObjectiveEnum"]

        dim = self.datasetX.shape[1]
        name = 'Feature optimization problem'
        M = 1  # M个目标函数
        maxormins = [1]  # 1表示最小化目标
        varTypes = [1] * dim  # 0: real; 1: integer
        lb = [0] * dim  # 决策变量边界
        ub = [1] * dim
        lbin = [1] * dim  # 决策变量边界那个值是否包含
        ubin = [1] * dim
        Problem.__init__(self, name, M, maxormins, dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        vars = pop.Phen
        objs = np.zeros(shape=(vars.shape[0], 1))
        popNum = vars.shape[0]
        for i in range(popNum):
            var = np.array(vars[i, :])
            if (var == 0).all():
                # 如果随机出来全部特征均不选取，则把目标值定义为无穷大
                loss = float("inf")
            else:
                featureCombinationParams = {
                    "isPerformCombination": True,
                    "featureFlags": var
                }

                featureCombination = FeatureCombination(self.datasetX, self.datasetY, self.featureNamesAfterCombination, featureCombinationParams)
                datasetAfterOptimizationX, datasetAfterOptimizationY, featureNamesAfterOptimization = featureCombination.getDatasetAfterCombination()
                dataset = Dataset(datasetAfterOptimizationX, datasetAfterOptimizationY, self.datasetParams)
                forecastTemplate = ForecastTemplate(dataset, self.forecastModelParams)
                loss = forecastTemplate.getObjective()
                # loss = np.random.random()
            processDraw = str(i+1)+"/"+str(popNum)+"\t["+("".join(["=" for _ in range(i)]))+">"+("".join(["." for _ in range(popNum-i-1)]))+"]-"+str(round((i+1)/popNum*100))+"%"
            print(processDraw)
            objs[i, 0] = loss
        pop.ObjV = objs
