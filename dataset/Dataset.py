# -*- coding: utf-8 -*-
# @File         : Dataset.py
# @Author       : Zhendong Zhang
# @Email        : zzd_zzd@hust.edu.cn
# @University   : Huazhong University of Science and Technology
# @Date         : 2019/9/7
# @Software     : PyCharm
# -*---------------------------------------------------------*-

import numpy as np
from enum import Enum
import json


class Dataset(object):

    def __init__(self, datasetX, datasetY, datasetParams):

        self.datasetParams = datasetParams
        self.datasetX = datasetX
        self.datasetY = datasetY
        self.isNormalize = datasetParams["isNormalize"]
        self.isFirstNormalize = datasetParams["isFirstNormalize"]
        self.normalizeType = datasetParams["normalizeType"]
        self.separateType = datasetParams["separateType"]
        self.separateValue = datasetParams["separateValue"]

        self.sampleNum = self.datasetX.shape[0]
        self.featureNum = self.datasetX.shape[1]

        # 合并特征和标签
        self.dataset = np.hstack((self.datasetX, self.datasetY))
        # 划分训练集和验证集
        self.trainX, self.trainY, self.validationX, self.validationY = self.__separate()
        self.trainSampleNum = self.trainX.shape[0]
        self.validSampleNum = self.validationX.shape[0]
        # 统计特征的统计值
        self.xMins = self.datasetX.min(axis=0)
        self.xMaxs = self.datasetX.max(axis=0)
        self.xMeans = self.datasetX.mean(axis=0)
        self.xSigs = self.datasetX.std(axis=0)
        self.xTrainMins = self.trainX.min(axis=0)
        self.xTrainMaxs = self.trainX.max(axis=0)
        self.xTrainMeans = self.trainX.mean(axis=0)
        self.xTrainSigs = self.trainX.std(axis=0)
        self.xValidMins = self.validationX.min(axis=0)
        self.xValidMaxs = self.validationX.max(axis=0)
        self.xValidMeans = self.validationX.mean(axis=0)
        self.xValidSigs = self.validationX.std(axis=0)
        # 统计标签的统计值
        self.yMins = self.datasetY.min(axis=0)
        self.yMaxs = self.datasetY.max(axis=0)
        self.yMeans = self.datasetY.mean(axis=0)
        self.ySigs = self.datasetY.std(axis=0)
        self.yTrainMins = self.trainY.min(axis=0)
        self.yTrainMaxs = self.trainY.max(axis=0)
        self.yTrainMeans = self.trainY.mean(axis=0)
        self.yTrainSigs = self.trainY.std(axis=0)
        self.yValidMins = self.validationY.min(axis=0)
        self.yValidMaxs = self.validationY.max(axis=0)
        self.yValidMeans = self.validationY.mean(axis=0)
        self.yValidSigs = self.validationY.std(axis=0)
        # 归一化
        self.trainX, self.trainY, self.validationX, self.validationY = self.__normalize()
        self.trainX2D, self.trainY2D, self.validationX2D, self.validationY2D = self.trainX, self.trainY, self.validationX, self.validationY

        # 保存验证集确定性预测结果
        self.validationD = None
        # 保存验证集概率预测结果
        self.validationP = None

    # 归一化
    def __normalize(self):
        if self.isNormalize:
            if self.isFirstNormalize:
                # 先归一化再划分数据集，这意味着基本统计值需要用整个数据集的
                trainX = self.__normalizeSingle(self.trainX, self.xMins, self.xMaxs, self.xMeans, self.xSigs)
                trainY = self.__normalizeSingle(self.trainY, self.yMins, self.yMaxs, self.yMeans, self.ySigs)
                validationX = self.__normalizeSingle(self.validationX, self.xMins, self.xMaxs, self.xMeans, self.xSigs)
                validationY = self.__normalizeSingle(self.validationY, self.yMins, self.yMaxs, self.yMeans, self.ySigs)
            else:
                # 先划分数据集后归一化，这意味着基本统计值采用训练集和验证集单独的
                trainX = self.__normalizeSingle(self.trainX, self.xTrainMins, self.xTrainMaxs, self.xTrainMeans,
                                                self.xTrainSigs)
                trainY = self.__normalizeSingle(self.trainY, self.yTrainMins, self.yTrainMaxs, self.yTrainMeans,
                                                self.yTrainSigs)
                validationX = self.__normalizeSingle(self.validationX, self.xValidMins, self.xValidMaxs,
                                                     self.xValidMeans,
                                                     self.xValidSigs)
                validationY = self.__normalizeSingle(self.validationY, self.yValidMins, self.yValidMaxs,
                                                     self.yValidMeans,
                                                     self.yValidSigs)
        else:
            trainX = self.trainX
            trainY = self.trainY
            validationX = self.validationX
            validationY = self.validationY
        return trainX, trainY, validationX, validationY

    def __normalizeSingle(self, series, mins, maxs, means, sigs):
        normSeries = np.zeros(shape=series.shape)
        for i in range(series.shape[1]):
            if self.normalizeType == 'NormalizeWithMinMax':
                normSeries[:, i] = (series[:, i] - mins[i]) / (maxs[i] - mins[i])
            elif self.normalizeType == 'NormalizeWithMeanSig':
                normSeries[:, i] = (series[:, i] - means[i]) / (sigs[i])
            else:
                raise ValueError("Dataset-->normalizeSingle()中normalizeType参数错误，只能是：[NormalizeWithMinMax, "
                                 "NormalizeWithMeanSig]\n")
        return normSeries

    def reverseLabel(self, normSeries):
        """
        归一化还原
        :param normSeries:
        :return:
        """
        if self.isNormalize:
            if self.isFirstNormalize:
                mins = self.yMins
                maxs = self.yMaxs
                means = self.yMeans
                sigs = self.ySigs
            else:
                mins = self.yValidMins
                maxs = self.yValidMaxs
                means = self.yValidMeans
                sigs = self.yValidSigs

            if self.normalizeType == 'NormalizeWithMinMax':
                series = mins + normSeries * (maxs - mins)
            elif self.normalizeType == 'NormalizeWithMeanSig':
                series = means + normSeries * sigs
            else:
                raise ValueError("Dataset-->reverseLabel()中normalizeType参数错误，只能是：[NormalizeWithMinMax, "
                                 "NormalizeWithMeanSig]\n")
        else:
            series = normSeries
        return series

    def __separate(self):
        """
        划分数据集
        :return:
        """
        separateType = self.separateType
        if separateType == "Ratio":
            sepIndex = round(self.sampleNum * self.separateValue)
        elif separateType == "Index":
            sepIndex = int(self.separateValue)
        elif separateType == "ValidationLength":
            sepIndex = self.sampleNum - int(self.separateValue)
        else:
            raise ValueError(
                "Dataset-->sepTrainAndValidation()中separateType参数错误，只能是：[Ratio, Index, ValidationLength]\n")

        trainX = self.datasetX[0:sepIndex, :]
        trainY = self.datasetY[0:sepIndex, :]
        validationX = self.datasetX[sepIndex::, :]
        validationY = self.datasetY[sepIndex::, :]
        return trainX, trainY, validationX, validationY

    def getDataset(self, dim):
        trainX = self.trainX2D
        trainY = self.trainY2D
        validationX = self.validationX2D
        validationY = self.validationY2D
        if dim == 3:
            trainX = trainX.reshape(self.trainSampleNum, 1, self.featureNum)
            trainY = trainY.reshape(self.trainSampleNum, 1, 1)
            validationX = validationX.reshape(self.validSampleNum, 1, self.featureNum)
            validationY = validationY.reshape(self.validSampleNum, 1, 1)
        self.trainX, self.trainY, self.validationX, self.validationY = trainX, trainY, validationX, validationY
        return trainX, trainY, validationX, validationY

    def setDataset(self, trainX, trainY, validationX, validationY):
        """
        为了格式统一，这些数据都应该是numpy二维数组
        :param trainX:
        :param trainY:
        :param validationX:
        :param validationY:
        :return:
        """
        self.trainX2D = trainX
        self.trainY2D = trainY
        self.validationX2D = validationX
        self.validationY2D = validationY


class SmallDataFactory(object):
    """
    从整个输入的大数据集上得到用于实验的小数据集
    规定从np.array到np.array
    """

    @staticmethod
    def getSmallData(bigData, smallDataParams):
        """
        从大数据集上获取小数据集，索引含首不含尾，数据集长度所见即所得
        起止索引和长度根据操作枚举类型按需设置
        :param bigData: 原始大数据集
        :param smallDataParams: 输入参数的dict
                    operateEnum: 从大到小的操作枚举
                    startIndex: 起始索引
                    endIndex: 结束索引
                    length: 小数据集长度
        :return: 小数据集
        """
        operateEnum = smallDataParams["operateEnum"]
        startIndex = smallDataParams["startIndex"]
        endIndex = smallDataParams["endIndex"]
        length = smallDataParams["length"]

        smallData = None
        if operateEnum == SmallDataOperateEnum.StartAndEnd:
            smallData = bigData[startIndex:endIndex]
        elif operateEnum == SmallDataOperateEnum.StartAndLength:
            endIndex = startIndex + length
            smallData = bigData[startIndex:endIndex]
        elif operateEnum == SmallDataOperateEnum.EndAndLength:
            startIndex = endIndex - length
            if startIndex < 0:
                startIndex = 0
            smallData = bigData[startIndex:endIndex]
        elif operateEnum == SmallDataOperateEnum.All:
            smallData = bigData
        else:
            raise ValueError("SmallDataFactory-->getSmallData()中operateEnum参数错误，只能是：\n" + (
                "\n".join(['%s:%s' % item for item in SmallDataOperateEnum.__dict__["_member_map_"].items()])))
        return smallData

    def optimizeDataset(self):
        """
        根据no free lunch理论，没有一个算法在所有数据集上都是最优的
        有时需要在少数数据集上侧重展示某些算法的性能，这个方法可帮助单个算法在大数据集上切分出较优该算法的小数据集
        暂时不想实现，有需要的时候再实现吧
        :return:
        """
        pass


class SmallDataOperateEnum(Enum):
    """
    将整个大数据集得到小数据集的操作枚举
    """

    # 通过起止索引确定，含首不含尾
    StartAndEnd = 1
    # 通过起始索引和数据集长度确定，含首定数据集长度
    StartAndLength = 2
    # 通过终止索引和数据集长度确定，不含尾定数据集长度
    EndAndLength = 3
    # 所有数据集
    All = 4


if __name__ == '__main__':
    bigData = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    startIndex = 0
    endIndex = 5
    smallData = SmallDataFactory.getSmallData(bigData, operateEnum=SmallDataOperateEnum.StartAndEnd,
                                              startIndex=startIndex,
                                              endIndex=endIndex)

    print(smallData)

    # datapath = '../resources/1518.xlsx'
    # sheetname = 'data'
    # data = DatasetUtils.readExcelData(datapath=datapath, sheetname=sheetname)
    # orgData = data[200:478][['Qi', 'Qo', 'Zu', 'H', 'Zd']]
    # featureIndex = np.array([0, 0, 1, 2, 2, 2, 3, 4, 4])
    # featureFlag = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1])
    # dataset = Dataset(orgData.values, featureIndex, featureFlag, 0.8)
