# -*- coding: utf-8 -*-
# @File         : OptimizationObjective.py
# @Author       : Zhendong Zhang
# @Organization : Huazhong University of Science and Technology
# @Email        : zzd_zzd@hust.edu.cn
# @Time         : 2020/7/2 19:01
from enum import Enum

from metric.Metric import ForecastPerformanceMetricEnum, ForecastPerformanceMetricFactory
from output.ResultsOutput import ResultsOutput


class OptimizationObjectiveFactory(object):

    @staticmethod
    def getObjective(forecastModel, optimizationObjectiveEnum):
        if optimizationObjectiveEnum == OptimizationObjectiveEnum.MinimizeDeterministicForecastingLoss:
            forecastModel.predict(isFlatten=True)
            dataset = forecastModel.dataset
            # 加负号是因为R2是越大，误差损失越小
            loss = -ForecastPerformanceMetricFactory.getPointForecastMetric(dataset,
                                                                            ForecastPerformanceMetricEnum.PointForecastMetricR2)
        elif optimizationObjectiveEnum == OptimizationObjectiveEnum.MaximizeProbabilisticForecastingReliability:
            forecastModel.getProbabilisticResults()
            dataset = forecastModel.dataset
            # 可靠性中，PIT值服从均匀分布的检测中P值越大越可靠
            loss = -ForecastPerformanceMetricFactory.getProbabilisticForecastMetric(dataset,
                                                                                    ForecastPerformanceMetricEnum.ProbabilisticForecastMetricPIT)
        else:
            raise ValueError(
                "OptimizationObjectiveFactory-->getObjective()中hyperParametersObjectiveEnum参数错误，只能是：\n" + (
                    "\n".join(
                        ['%s:%s' % item for item in OptimizationObjectiveEnum.__dict__["_member_map_"].items()])))
        # 把整个运行过程中最优的模型实时比较，实时保存
        isUpdate = ResultsOutput.pickleFinalForecastModel(loss, forecastModel)
        return loss, isUpdate


class OptimizationObjectiveEnum(Enum):
    """
    超参数优化和特征优化目标枚举，强制超参数和特征优化时目标要一致
    """

    # 最小化确定性预报损失，也即点预测精度最高
    MinimizeDeterministicForecastingLoss = 1
    # 最大化概率预报可靠性
    MaximizeProbabilisticForecastingReliability = 2
