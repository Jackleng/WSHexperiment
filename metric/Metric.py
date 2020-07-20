# -*- coding: utf-8 -*-
# @File         : Metric.py
# @Author       : Zhendong Zhang
# @Organization : Huazhong University of Science and Technology
# @Email        : zzd_zzd@hust.edu.cn
# @Time         : 2020/6/30 14:52
from enum import Enum
import scipy.stats as stats
from minepy import MINE
import numpy as np

import forecast
from util.math.MathUtils import MathInterpolateUtils


class CorrelationAnalysisMetricFactory(object):

    @staticmethod
    def getMetric(x, y, metricEnum):
        metric = None
        x = x.flatten()
        y = y.flatten()
        if metricEnum == CorrelationAnalysisMetricEnum.PCC:
            metric = CorrelationAnalysisMetricFactory.__calMetricPCC(x, y)
        elif metricEnum == CorrelationAnalysisMetricEnum.MIC:
            metric = CorrelationAnalysisMetricFactory.__calMetricMIC(x, y)
        else:
            raise ValueError("CorrelationAnalysisMetricFactory-->getMetric()中metricEnum参数错误，只能是：\n" + ("\n".join(
                ['%s:%s' % item for item in CorrelationAnalysisMetricEnum.__dict__["_member_map_"].items()])))
        return metric

    @staticmethod
    def __calMetricPCC(x, y):
        # PCC是皮尔逊相关系数，p是显著性检验，越小越显著
        (PCC, p) = stats.pearsonr(x, y)
        return abs(PCC)

    @staticmethod
    def __calMetricMIC(x, y):
        # 最大信息系数
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(x, y)
        return mine.mic()


class CorrelationAnalysisMetricEnum(Enum):
    """
    相关分析指标
    """

    # 皮尔逊相关系数的绝对值
    PCC = 1
    # 最大信息系数
    MIC = 2


class ForecastPerformanceMetricFactory(object):
    @staticmethod
    def getPointForecastMetric(dataset, metricEnum):
        metric = None

        predictions = dataset.validationD
        predictions = dataset.reverseLabel(predictions)
        predictions = predictions.flatten()
        observations = dataset.validationY
        observations = dataset.reverseLabel(observations)
        observations = observations.flatten()

        if metricEnum == ForecastPerformanceMetricEnum.PointForecastMetricMAE:
            metric = ForecastPerformanceMetricFactory.__calMetricMAE(predictions, observations)
        elif metricEnum == ForecastPerformanceMetricEnum.PointForecastMetricMAPE:
            metric = ForecastPerformanceMetricFactory.__calMetricMAPE(predictions, observations)
        elif metricEnum == ForecastPerformanceMetricEnum.PointForecastMetricMSE:
            metric = ForecastPerformanceMetricFactory.__calMetricMSE(predictions, observations)
        elif metricEnum == ForecastPerformanceMetricEnum.PointForecastMetricR2:
            metric = ForecastPerformanceMetricFactory.__calMetricR2(predictions, observations)
        elif metricEnum == ForecastPerformanceMetricEnum.PointForecastMetricRMSE:
            metric = ForecastPerformanceMetricFactory.__calMetricRMSE(predictions, observations)
        else:
            raise ValueError("ForecastPerformanceMetricFactory-->getPointForecastMetric()方法中metricEnum参数只能是："
                             "[PointForecastMetricMAE, PointForecastMetricMAPE, PointForecastMetricMSE, "
                             "PointForecastMetricR2 ,PointForecastMetricRMSE]")

        return metric

    @staticmethod
    def getIntervalForecastMetric(dataset, metricEnum, alpha=0.9):
        metric = None

        observations = dataset.validationY.flatten()
        observations = dataset.reverseLabel(observations)
        observations = observations.flatten()

        validationP = dataset.validationP

        lowerAlpha = (1 - alpha) / 2
        upperAlpha = 1 - lowerAlpha

        lower = forecast.Forecast.ForecastModelBase.getPredictionsByQuantile(validationP, lowerAlpha)
        upper = forecast.Forecast.ForecastModelBase.getPredictionsByQuantile(validationP, upperAlpha)
        lower = lower.flatten()
        upper = upper.flatten()
        if metricEnum == ForecastPerformanceMetricEnum.IntervalForecastMetricCP:
            metric = ForecastPerformanceMetricFactory.__calMetricCP(lower, upper, observations)
        elif metricEnum == ForecastPerformanceMetricEnum.IntervalForecastMetricMWP:
            metric = ForecastPerformanceMetricFactory.__calMetricMWP(lower, upper, observations)
        elif metricEnum == ForecastPerformanceMetricEnum.IntervalForecastMetricCM:
            metric = ForecastPerformanceMetricFactory.__calMetricCM(lower, upper, observations)
        elif metricEnum == ForecastPerformanceMetricEnum.IntervalForecastMetricMC:
            metric = ForecastPerformanceMetricFactory.__calMetricMC(lower, upper, observations)
        else:
            raise ValueError("ForecastPerformanceMetricFactory-->getIntervalForecastMetric()方法中metricEnum参数只能是："
                             "[IntervalForecastMetricCP, IntervalForecastMetricMWP, IntervalForecastMetricCM, "
                             "IntervalForecastMetricMC]")
        return metric

    @staticmethod
    def getProbabilisticForecastMetric(dataset, metricEnum):
        metric = None
        # 概率预报中的指标均采用归一化的值计算，将预报值还原再计算太麻烦了
        observations = dataset.validationY.flatten()
        observations = dataset.reverseLabel(observations)
        observations = observations.flatten()

        validationP = dataset.validationP
        cdfs = validationP["cdfs"]

        if metricEnum == ForecastPerformanceMetricEnum.ProbabilisticForecastMetricCRPS:
            metric = ForecastPerformanceMetricFactory.__calMetricCRPS(cdfs, observations)
        elif metricEnum == ForecastPerformanceMetricEnum.ProbabilisticForecastMetricPIT:
            metric = ForecastPerformanceMetricFactory.__calMetricPIT(cdfs, observations)
        else:
            raise ValueError("ForecastPerformanceMetricFactory-->getProbabilisticForecastMetric()方法中metricEnum参数只能是："
                             "[ProbabilisticForecastMetricCRPS, ProbabilisticForecastMetricPIT]")

        return metric

    @staticmethod
    def __calMetricMAE(predictions, observations):
        MAE = np.mean(np.abs(predictions - observations))
        return MAE

    @staticmethod
    def __calMetricMSE(predictions, observations):
        MSE = np.mean(np.power(predictions - observations, 2))
        return MSE

    @staticmethod
    def __calMetricRMSE(predictions, observations):
        MSE = ForecastPerformanceMetricFactory.__calMetricMSE(predictions, observations)
        RMSE = np.sqrt(MSE)
        return RMSE

    @staticmethod
    def __calMetricMAPE(predictions, observations):
        # 当observations中有0时，这个指标有除0风险
        try:
            MAPE = np.mean(np.true_divide(np.abs(predictions - observations), np.abs(observations)))
        except Exception as e:
            MAPE = np.nan
        return MAPE

    @staticmethod
    def __calMetricR2(predictions, observations):
        mean = np.mean(observations)
        numerator = np.sum(np.power(observations - predictions, 2))
        denominator = np.sum(np.power(observations - mean, 2))
        R2 = 1 - numerator / denominator
        return R2

    @staticmethod
    def __calMetricCP(lower, upper, observations):
        N = observations.shape[0]
        count = 0
        for i in range(N):
            if lower[i] <= observations[i] <= upper[i]:
                count = count + 1
        CP = count / N
        return CP

    @staticmethod
    def __calMetricMWP(lower, upper, observations):
        N = observations.shape[0]
        MWP = 0
        for i in range(N):
            if upper[i] < lower[i]:
                print(i)
            MWP = MWP + (upper[i] - lower[i]) / np.abs(observations[i])
        MWP = MWP / N
        return MWP

    @staticmethod
    def __calMetricCM(lower, upper, observations):
        CM = ForecastPerformanceMetricFactory.__calMetricCP(lower, upper,
                                                            observations) / ForecastPerformanceMetricFactory.__calMetricMWP(
            lower, upper, observations)
        return CM

    @staticmethod
    def __calMetricMC(lower, upper, observations):
        MC = 1.0 / ForecastPerformanceMetricFactory.__calMetricCM(lower, upper, observations)
        return MC

    @staticmethod
    def __calMetricCRPS(cdfs, observations):
        num = len(observations)
        areas = np.zeros(shape=(num, 1))
        for i in range(num):
            cdf = cdfs[i]
            x = cdf["x"]
            F = cdf["F"]

            y = np.zeros(shape=x.shape)
            area = 0.0
            for j in range(1, len(x)-1):
                y[j] = F[j] - ForecastPerformanceMetricFactory.getH(x[j], observations[i])
                y[j] = np.power(y[j], 2)
                if j >= 2:
                    area = area + (y[j] + y[j - 1]) * (x[j] - x[j - 1]) / 2
            areas[i, 0] = area
        CRPS = np.mean(areas)
        return CRPS

    @staticmethod
    def getH(prediction, observation):
        if prediction < observation:
            return 0
        else:
            return 1

    @staticmethod
    def getPIT(cdfs, observations):
        num = len(observations)
        pit = np.zeros(shape=(num, 1))
        for i in range(num):
            cdf = cdfs[i]
            x = cdf["x"]
            F = cdf["F"]
            observation = observations[i]
            pit[i] = MathInterpolateUtils.interp1d(x, F, observation, kind="slinear")
        return pit

    @staticmethod
    def __calMetricPIT(cdfs, observations):
        pit = ForecastPerformanceMetricFactory.getPIT(cdfs, observations)
        D, P = stats.kstest(pit, cdf='uniform')
        # if P > 0.05:
        #     print('PIT值均匀分布检验P=', P, '>0.05, 预测结果可靠！')
        # else:
        #     print('PIT值均匀分布检验P=', P, '<=0.05, 预测结果不可靠！')

        return P


class ForecastPerformanceMetricEnum(Enum):
    """
    预报性能指标
    """

    PointForecastMetricMAE = 1
    PointForecastMetricMSE = 3
    PointForecastMetricRMSE = 4
    PointForecastMetricMAPE = 5
    PointForecastMetricR2 = 6

    IntervalForecastMetricCP = 11
    IntervalForecastMetricMWP = 12
    IntervalForecastMetricCM = 13
    IntervalForecastMetricMC = 14

    ProbabilisticForecastMetricCRPS = 21
    ProbabilisticForecastMetricPIT = 22
