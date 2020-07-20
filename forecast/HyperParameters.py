# -*- coding: utf-8 -*-
# @File         : HyperParameters.py
# @Author       : Zhendong Zhang
# @Organization : Huazhong University of Science and Technology
# @Email        : zzd_zzd@hust.edu.cn
# @Time         : 2020/7/1 19:37
from enum import Enum

from hyperopt import STATUS_OK, Trials, tpe, fmin

from forecast.OptimizationObjective import OptimizationObjectiveFactory
from output.ResultsOutput import ResultsOutput
from util.const.GlobalConstants import GlobalConstants


class HyperParametersOptimizationTemplate(object):

    def __init__(self, forecastModel, optimizationObjectiveEnum):
        self.forecastModel = forecastModel
        self.optimizationObjectiveEnum = optimizationObjectiveEnum
        self.trials = Trials()
        # 判断此次超参数优化有没有替换到当前的最优解，如果有这次就是运行过程中的当前最优超参数优化过程，应该被保存
        self.isUpdate = False
        self.best = None
        self.optimize()

    def getObjective(self, optimizedHyperParameters=None):
        """
        在当前超参数下，优化的目标值是多少，全部转换为最小化
        :param optimizedHyperParameters:
        :return:
        """
        # import numpy as np
        # loss = np.random.random()
        # isUpdate = True
        forecastModel = self.forecastModel
        forecastModel.setOptimizedHyperParameters(optimizedHyperParameters)
        forecastModel.run()
        loss, isUpdate = OptimizationObjectiveFactory.getObjective(forecastModel, self.optimizationObjectiveEnum)
        if isUpdate:
            self.isUpdate = True
        return {'loss': loss, 'params': optimizedHyperParameters, 'status': STATUS_OK}

    def getBestObjective(self):
        """获取整个超参数优化过程中的最优目标值"""
        return self.trials.best_trial['result']["loss"]

    def getHyperParametersOptimizationProcess(self):
        hyperParametersOptimizationProcess = {
            "best": self.best,
            "trials": self.trials,
        }
        return hyperParametersOptimizationProcess

    def getOptimalHyperParameters(self):
        """
        获取最优的超参数
        :return:
        """
        bestTrial = self.trials.best_trial['result']
        optimalHyperParameters = bestTrial['params']
        return optimalHyperParameters

    def optimize(self):
        space = self.forecastModel.getOptimizedHyperParametersRange()
        self.best = fmin(self.getObjective, space=space, algo=tpe.suggest, max_evals=3, trials=self.trials)
        if self.isUpdate:
            ResultsOutput.pickleFinalHyperParametersOptimizationProcess(self.getHyperParametersOptimizationProcess())
        return self.best, self.trials


