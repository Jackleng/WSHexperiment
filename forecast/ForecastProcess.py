# -*- coding: utf-8 -*-
# @File         : ForecastProcess.py
# @Author       : Zhendong Zhang
# @Organization : Huazhong University of Science and Technology
# @Email        : zzd_zzd@hust.edu.cn
# @Time         : 2020/7/2 19:22
from dataset.Dataset import Dataset
from forecast.Feature import FeatureTemplate
from forecast.Forecast import ForecastTemplate


class ForecastProcess(object):
    def __init__(self, orgData, featureParams, datasetParams, forecastModelParams):
        self.orgData = orgData
        self.featureParams = featureParams
        self.datasetParams = datasetParams
        self.forecastModelParams = forecastModelParams
        
        # 完成 特征生成-->特征选择-->特征组合-->特征优化
        featureTemplate = FeatureTemplate(orgData, featureParams, datasetParams, forecastModelParams)
        self.datasetX, self.datasetY = featureTemplate.getFinalDataset()
        # 构造数据集
        self.dataset = Dataset(self.datasetX, self.datasetY, datasetParams)
        # 完成 超参数优化-->模型训练
        forecastTemplate = ForecastTemplate(self.dataset, forecastModelParams)
        self.forecastModel = forecastTemplate.getFinalForecastModel()
        # 完成最终的预报
        self.forecastModel.predict()
        self.forecastModel.getProbabilisticResults()
