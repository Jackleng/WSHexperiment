# -*- coding: utf-8 -*-
# @File         : Forecast.py
# @Author       : Zhendong Zhang
# @Organization : Huazhong University of Science and Technology
# @Email        : zzd_zzd@hust.edu.cn
# @Time         : 2020/6/30 18:52
import abc
import math
from abc import ABC

from keras import Sequential, regularizers
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM
from hyperopt import hp
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, Matern as M
from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg
from enum import Enum
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from forecast.HyperParameters import HyperParametersOptimizationTemplate
from forecast.OptimizationObjective import OptimizationObjectiveFactory
from forecast.recurrent import LSTM, GRU, SWLSTM
from util.math.MathUtils import MathInterpolateUtils


class ForecastTemplate(object):
    def __init__(self, dataset, forecastModelParams):
        self.dataset = dataset
        self.forecastModelParams = forecastModelParams
        self.hyperParametersOptimizationProcess = None
        self.loss = None

        # 获取预报模型
        self.forecastModelEnum = forecastModelParams["forecastModelEnum"]
        isPerformOptimization = forecastModelParams["isPerformOptimization"]
        isUsingDefaultFixedHyperParameters = forecastModelParams["isUsingDefaultFixedHyperParameters"]
        if isUsingDefaultFixedHyperParameters:
            self.fixedHyperParameters = None
        else:
            self.fixedHyperParameters = forecastModelParams["fixedHyperParameters"]
        forecastModel = ForecastModelFactory.getForecastModel(self.forecastModelEnum, dataset,
                                                              fixedHyperParameters=self.fixedHyperParameters)
        if isPerformOptimization:
            optimizationObjectiveEnum = forecastModelParams["optimizationObjectiveEnum"]
            hyperParametersOptimization = HyperParametersOptimizationTemplate(forecastModel,
                                                                              optimizationObjectiveEnum)
            self.loss = hyperParametersOptimization.getBestObjective()
            self.optimizedHyperParameters = hyperParametersOptimization.getOptimalHyperParameters()
        else:
            isUsingDefaultOptimizedHyperParameters = forecastModelParams["isUsingDefaultOptimizedHyperParameters"]
            if isUsingDefaultOptimizedHyperParameters:
                self.optimizedHyperParameters = None
            else:
                self.optimizedHyperParameters = forecastModelParams["optimizedHyperParameters"]

            forecastModel = ForecastModelFactory.getForecastModel(self.forecastModelEnum, dataset,
                                                                  fixedHyperParameters=self.fixedHyperParameters,
                                                                  optimizedHyperParameters=self.optimizedHyperParameters)
            forecastModel.run()
            optimizationObjectiveEnum = forecastModelParams["optimizationObjectiveEnum"]
            self.loss, _ = OptimizationObjectiveFactory.getObjective(forecastModel, optimizationObjectiveEnum)

    def getObjective(self):
        return self.loss

    def getFinalForecastModel(self):
        """
        谨慎使用这个方法，因为即使是同样的参数对于神经网络模型而言重新训练得到的模型权重也会不一样
        建议只在GPR，QR和SVR这种超参数一样得到的结果就一样的情况下用
        对于神经网络这类模型的forecastModel已经在训练过程中持久化了，不用重新训练，这样保证结果的一致性
        :return:
        """
        forecastModel = ForecastModelFactory.getForecastModel(self.forecastModelEnum, self.dataset,
                                                              fixedHyperParameters=self.fixedHyperParameters,
                                                              optimizedHyperParameters=self.optimizedHyperParameters)
        forecastModel.run()
        return forecastModel


class ForecastModelFactory(object):
    @staticmethod
    def getForecastModel(forecastModelEnum, dataset, optimizedHyperParameters=None, fixedHyperParameters=None):
        if forecastModelEnum == ForecastModelEnum.ForecastModelLSTM:
            forecastModel = ForecastModelLSTM(dataset, optimizedHyperParameters, fixedHyperParameters)
        elif forecastModelEnum == ForecastModelEnum.ForecastModelGRU:
            forecastModel = ForecastModelGRU(dataset, optimizedHyperParameters, fixedHyperParameters)
        elif forecastModelEnum == ForecastModelEnum.ForecastModelSWLSTM:
            forecastModel = ForecastModelSWLSTM(dataset, optimizedHyperParameters, fixedHyperParameters)
        elif forecastModelEnum == ForecastModelEnum.ForecastModelCNN:
            forecastModel = ForecastModelCNN(dataset, optimizedHyperParameters, fixedHyperParameters)
        elif forecastModelEnum == ForecastModelEnum.ForecastModelANN:
            forecastModel = ForecastModelANN(dataset, optimizedHyperParameters, fixedHyperParameters)
        elif forecastModelEnum == ForecastModelEnum.ForecastModelSVR:
            forecastModel = ForecastModelSVR(dataset, optimizedHyperParameters, fixedHyperParameters)
        elif forecastModelEnum == ForecastModelEnum.ForecastModelGPR:
            forecastModel = ForecastModelGPR(dataset, optimizedHyperParameters, fixedHyperParameters)
        elif forecastModelEnum == ForecastModelEnum.ForecastModelQR:
            forecastModel = ForecastModelQR(dataset, optimizedHyperParameters, fixedHyperParameters)
        else:
            raise ValueError("ForecastModelFactory-->getForecastModel()中forecastModelEnum参数错误，只能是：\n" + ("\n".join(
                ['%s:%s' % item for item in ForecastModelEnum.__dict__["_member_map_"].items()])))
        return forecastModel


class ForecastModelEnum(Enum):
    """
    预报模型枚举
    """
    ForecastModelLSTM = 1
    ForecastModelGRU = 2
    ForecastModelSWLSTM = 3
    ForecastModelCNN = 4
    ForecastModelANN = 5
    ForecastModelSVR = 6
    ForecastModelGPR = 7
    ForecastModelQR = 8


class ForecastModelBase(object):

    def __init__(self, dataset, optimizedHyperParameters=None, fixedHyperParameters=None):
        self.dataset = dataset
        if optimizedHyperParameters is None:
            self.optimizedHyperParameters = self.getDefaultOptimizedHyperParameters()
        else:
            self.optimizedHyperParameters = optimizedHyperParameters

        if fixedHyperParameters is None:
            self.fixedHyperParameters = self.getDefaultFixedHyperParameters()
        else:
            self.fixedHyperParameters = fixedHyperParameters
        self.model = None

    def setOptimizedHyperParameters(self, optimizedHyperParameters):
        self.optimizedHyperParameters = optimizedHyperParameters

    def getOptimizedHyperParameters(self):
        return self.optimizedHyperParameters

    def getFixedHyperParameters(self):
        return self.fixedHyperParameters

    def run(self):
        """
        启动整套模型
        :return:
        """
        # 构造模型
        self.constructModel()
        # 训练模型
        self.fit()

    @abc.abstractmethod
    def constructModel(self):
        """
        根据具体的预报模型及对应超参数来构造
        :return:
        """

    @abc.abstractmethod
    def getDefaultOptimizedHyperParameters(self):
        """
        内置一套默认的待优化超参数
        :return:
        """

    @abc.abstractmethod
    def getDefaultFixedHyperParameters(self):
        """
        内置一套默认的不参与优化的超参数
        :return:
        """

    @abc.abstractmethod
    def getOptimizedHyperParametersRange(self):
        """
        待优化超参数范围
        :return:
        """

    @abc.abstractmethod
    def fit(self):
        """
        训练模型
        :return:
        """
        pass

    def predict(self, validationX=None, isFlatten=False):
        """
        确定性预报结果
        :param validationX: 待预报数据集的特征
        :param isFlatten: 是否把预测结果撸直，撸直之后可以直接用来算指标之类的
        :return:
            predictions: 确定性预报结果，也是概率结果的预测均值
        """
        if validationX is None:
            validationX = self.dataset.validationX
        predictions = self.model.predict(validationX)
        if isFlatten:
            predictions = predictions.flatten()
        self.dataset.validationD = predictions
        return predictions

    def getProbabilisticResults(self, probabilisticForecastModelParams, validationX=None):
        """
        获取概率预报结果
        这里设计成PDF和CDF均不去关心具体的函数关系式，而采用非常细粒度的置信度数组得到对应的pdfs和cdfs
        :param probabilisticForecastModelParams: 概率预报模型参数
        :param validationX: 待预报数据集的特征
        :return:
            pdfs: 非常细粒度的概率密度函数散点, numpy一维数组 pdfs[i]:第i个样本点；每个元素是一个dict,x:预测数据，f:概率密度值
            cdfs: 非常细粒度的累计分布函数散点, 与pdfs对应，x:预测数据，F:累计分布值
        """
        if validationX is None:
            validationX = self.dataset.validationX
        newTrainX = self.predict(self.dataset.trainX, isFlatten=True)
        newTrainX = newTrainX.reshape(len(newTrainX), 1)
        newTrainY = self.dataset.trainY.flatten()
        newTrainY = newTrainY.reshape(len(newTrainY), 1)
        newValidationX = self.predict(self.dataset.validationX, isFlatten=True)
        newValidationX = newValidationX.reshape(len(newValidationX), 1)
        newValidationY = self.dataset.validationY.flatten()
        newValidationY = newValidationY.reshape(len(newValidationY), 1)
        newDataset = self.dataset
        newDataset.setDataset(newTrainX, newTrainY, newValidationX, newValidationY)

        forecastTemplate = ForecastTemplate(newDataset, probabilisticForecastModelParams)
        probabilisticModel = forecastTemplate.getFinalForecastModel()
        probabilisticResults = probabilisticModel.getProbabilisticResults(None)
        self.dataset.validationP = probabilisticResults
        return probabilisticResults

    @staticmethod
    def getPredictionsByQuantile(probabilisticResults, quantile=0.5):
        cdfs = probabilisticResults["cdfs"]
        sampleNum = cdfs.shape[0]
        quantilePredictions = np.zeros(shape=(sampleNum, 1))
        for i in range(sampleNum):
            cdf = cdfs[i]
            F = cdf["F"]
            x = cdf["x"]
            quantilePredictions[i, 0] = MathInterpolateUtils.interp1d(F, x, quantile, kind="slinear")
        quantilePredictions = quantilePredictions.flatten()
        return quantilePredictions


class ForecastModelNNBase(ForecastModelBase, ABC):
    """
    神经网络类模型的基类\n
    主要用于对比的公平性，将默认参数和超参数优化范围限定一致
    对于个别模型超参数超过以下范围的可以重写这些方法
    """
    def fit(self):
        optimizedHyperParameters = self.optimizedHyperParameters
        fixedHyperParameters = self.fixedHyperParameters

        batchSize = optimizedHyperParameters["batchSize"]
        epochs = fixedHyperParameters["epochs"]

        reduceLr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=10,
                                     verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.0001)
        trainX, trainY, validationX, validationY = self.dataset.getDataset(3)
        self.model.fit(trainX, trainY, batch_size=batchSize, epochs=epochs, callbacks=[reduceLr], shuffle=False)

    def getOptimizedHyperParametersRange(self):
        optimizedHyperParametersRange = {
            "hiddenLayerNum": hp.choice("hiddenLayerNum", [1, 2, 3]),
            "hiddenLayerNodeNum": hp.choice("hiddenLayerNodeNum", [2, 4, 8, 16, 32, 64]),
            "dropoutRate": hp.loguniform('dropoutRate', np.log(0.0001), np.log(0.001)),
            'batchSize': hp.choice('batchSize', [8, 16, 32, 64]),
            "regularizeRate": hp.loguniform('regularizeRate', np.log(0.0001), np.log(0.001)),
        }
        return optimizedHyperParametersRange

    def getDefaultOptimizedHyperParameters(self):
        optimizedHyperParameters = dict()
        # 隐藏层数
        optimizedHyperParameters["hiddenLayerNum"] = 1
        # 隐藏层节点数
        optimizedHyperParameters["hiddenLayerNodeNum"] = 4
        # dropoutRate
        optimizedHyperParameters["dropoutRate"] = 0.001
        # batchSize
        optimizedHyperParameters["batchSize"] = 32
        # L1_L2 regularizeRate
        optimizedHyperParameters["regularizeRate"] = 0.001
        return optimizedHyperParameters

    def getDefaultFixedHyperParameters(self):
        fixedHyperParameters = dict()

        # 训练过程中采用的损失函数
        fixedHyperParameters["lossNameInTraining"] = "mse"
        # 采用什么优化器来训练
        fixedHyperParameters["optimizer"] = "adam"
        # 训练轮数
        fixedHyperParameters["epochs"] = 200
        return fixedHyperParameters


class ForecastModelLSTM(ForecastModelNNBase):
    """
    LSTM预报模型
    """

    def constructModel(self):
        optimizedHyperParameters = self.optimizedHyperParameters
        fixedHyperParameters = self.fixedHyperParameters

        hiddenLayerNum = optimizedHyperParameters["hiddenLayerNum"]
        hiddenLayerNodeNum = optimizedHyperParameters["hiddenLayerNodeNum"]
        dropoutRate = optimizedHyperParameters["dropoutRate"]
        regularizeRate = optimizedHyperParameters["regularizeRate"]

        lossNameInTraining = fixedHyperParameters["lossNameInTraining"]
        optimizer = fixedHyperParameters["optimizer"]

        regularize = regularizers.l1_l2(regularizeRate, regularizeRate)

        model = Sequential()
        for i in range(hiddenLayerNum):
            model.add(LSTM(units=hiddenLayerNodeNum, recurrent_regularizer=regularize, activity_regularizer=regularize,
                           bias_regularizer=regularize, return_sequences=True))
            model.add(Dropout(dropoutRate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss=lossNameInTraining, optimizer=optimizer)
        self.model = model


class ForecastModelGPR(ForecastModelBase):
    """
    GPR预报模型
    """

    def constructModel(self):
        optimizedHyperParameters = self.optimizedHyperParameters
        fixedHyperParameters = self.fixedHyperParameters

        kernelName = optimizedHyperParameters["kernelName"]

        if kernelName == 'RBF':
            kernel = C(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-5, 1e5))
        elif kernelName == 'Matern':
            kernel = M(1.0, (1e-5, 1e5), nu=1.5)
        elif kernelName == 'RationalQuadratic':
            kernel = RQ(1.0, 1.0, (1e-5, 1e5), (1e-5, 1e5))

        model = GaussianProcessRegressor(kernel=kernel)

        self.model = model

    def fit(self):
        trainX, trainY, validationX, validationY = self.dataset.getDataset(2)
        self.model.fit(trainX, trainY)

    def getOptimizedHyperParametersRange(self):
        optimizedHyperParametersRange = {
            "kernelName": hp.choice("kernelName", ['RBF', 'Matern', 'RationalQuadratic']),
        }
        return optimizedHyperParametersRange

    def getDefaultOptimizedHyperParameters(self):
        optimizedHyperParameters = dict()
        # 核函数名称
        optimizedHyperParameters["kernelName"] = "RationalQuadratic"
        return optimizedHyperParameters

    def getDefaultFixedHyperParameters(self):
        fixedHyperParameters = dict()
        return fixedHyperParameters

    def getProbabilisticResults(self, probabilisticForecastModelParams, validationX=None):
        if validationX is None:
            validationX = self.dataset.validationX
        predictions, sigmas = self.model.predict(validationX, return_std=True)
        predictions = predictions.flatten()
        sigmas = predictions.flatten()
        validSampleNum = len(predictions)
        pointNum = 1001
        pdfs = []
        cdfs = []
        for i in range(validSampleNum):
            u = predictions[i]
            s = sigmas[i]
            # # 默认含首含尾
            # x = np.linspace(u - 3 * s, u + 3 * s, pointNum)
            # f = stats.norm(u, s).pdf(x)
            #
            # p = dict()
            # p["x"] = x
            # p["f"] = f
            # pdfs.append(p)

            # 刚好从0到1步长0.001，也恰好是1001个点
            F = np.arange(0, 1.001, 0.001)
            F[0] = 0.0001
            F[-1] = 1 - 0.0001
            x = stats.norm(u, s).ppf(F)
            x = self.dataset.reverseLabel(x)
            c = dict()
            c["x"] = x
            c["F"] = F
            cdfs.append(c)

            # 方法1：面积定义来求，假设小矩形，这个过程中推荐方法1
            xNew = np.linspace(x.min(), x.max(), len(x))
            y = MathInterpolateUtils.interp1d(x, F, xNew, kind="slinear")
            f = np.zeros(shape=x.shape)
            for j in range(1, len(f)):
                f[j] = (y[j] - y[j - 1]) / (xNew[j] - xNew[j - 1])
            x = xNew

            p = dict()
            p["x"] = x
            p["f"] = f
            pdfs.append(p)
        probabilisticResults = {
            "pdfs": np.array(pdfs),
            "cdfs": np.array(cdfs)
        }
        self.dataset.validationP = probabilisticResults
        return probabilisticResults


class ForecastModelQR(ForecastModelBase):
    """
    QR预报模型
    """

    def constructModel(self):
        """
        QR比较特殊，无需构造模型，或者说它构造模型和训练是同时完成的，所以实现均在fit()方法中
        :return:
        """
        pass

    def fit(self):
        optimizedHyperParameters = self.optimizedHyperParameters
        fixedHyperParameters = self.fixedHyperParameters

        kernelName = optimizedHyperParameters["kernelName"]
        trainX, trainY, validationX, validationY = self.dataset.getDataset(2)
        self.model = QuantReg(trainY, trainX)

    def predict(self, validationX=None, isFlatten=False):
        if validationX is None:
            validationX = self.dataset.validationX
        optimizedHyperParameters = self.optimizedHyperParameters
        kernelName = optimizedHyperParameters["kernelName"]
        results = self.model.fit(q=0.5, kernel=kernelName)
        predictions = self.model.predict(params=results.params, exog=validationX)
        if isFlatten:
            predictions = predictions.flatten()
        self.dataset.validationD = predictions
        return predictions

    def getOptimizedHyperParametersRange(self):
        optimizedHyperParametersRange = {
            "kernelName": hp.choice("kernelName", ['epa', 'cos', 'gau', 'par']),
        }
        return optimizedHyperParametersRange

    def getDefaultOptimizedHyperParameters(self):
        optimizedHyperParameters = dict()
        # 核函数名称
        optimizedHyperParameters["kernelName"] = "epa"
        return optimizedHyperParameters

    def getDefaultFixedHyperParameters(self):
        fixedHyperParameters = dict()
        return fixedHyperParameters

    def getProbabilisticResults(self, probabilisticForecastModelParams, validationX=None):
        if validationX is None:
            validationX = self.dataset.validationX
        validSampleNum = validationX.shape[0]
        optimizedHyperParameters = self.optimizedHyperParameters
        kernelName = optimizedHyperParameters["kernelName"]

        # 刚好从0到1步长0.001，也恰好是1001个点
        F = np.arange(0, 1.001, 0.001)
        predictions = np.zeros(shape=(validSampleNum, len(F)))
        for i in range(len(F)):
            q = F[i]
            if 0 < q < 1:
                results = self.model.fit(q=q, kernel=kernelName)
                prediction = self.model.predict(params=results.params, exog=validationX)
                predictions[:, i] = prediction.T
        predictions[:, 0] = 2 * predictions[:, 1] - predictions[:, 2]
        predictions[:, -1] = 2 * predictions[:, -2] - predictions[:, -3]
        predictions.sort(axis=1)
        pdfs = []
        cdfs = []
        for i in range(validSampleNum):
            # 刚好从0到1步长0.001，也恰好是1001个点
            x = predictions[i, :]
            x = self.dataset.reverseLabel(x)
            c = dict()
            c["x"] = x
            c["F"] = F
            cdfs.append(c)

            # 已知概率密度函数PDF去求累计分布函数CDF，这是确定的过程
            # 已知CDF反求PDF，在PDF形式未知的情况下，根据所求方法采用的假设不同得到的PDF不同
            # 用面积定义来求，假设在散点很密的情况下，可以简化为小梯形面积或者小矩形面积，但这个假设不同会导致PDF形式差别很大
            # 也可以根据CDF分布来随机生成很多样本，再采用核密度估计方法也能得到PDF，总之取决于假设

            # 方法1：面积定义来求，假设小矩形，这个过程中推荐方法1
            xNew = np.linspace(x.min(), x.max(), len(x))
            y = MathInterpolateUtils.interp1d(x, F, xNew, kind="slinear")
            f = np.zeros(shape=x.shape)
            for j in range(1, len(f)):
                f[j] = (y[j] - y[j - 1]) / (xNew[j] - xNew[j - 1])
            x = xNew

            # 方法2：面积定义法，假设小梯形
            # f = np.zeros(shape=x.shape)
            # for j in range(1, len(F)):
            #     f[j] = 2 * (F[j] - F[j - 1]) / (x[j] - x[j - 1]) - f[j - 1]

            # 方法3：核密度估计
            # 首先需要针对CDF产生均匀分布的随机数，由于计算过程中分位数已经是均匀分布的了，所以可以直接对对应的x值进行估计
            # 方法3很费时，除了展示个别时段的PDF，整个过程中基本都在用CDF而不是PDF，所以在这个过程中不建议采用方法3
            # 只在专门展示PDF的服务里使用这个方法
            # paramGrid = {'bandwidth': np.arange(0, 5, 0.5)}
            # kde = KernelDensity(kernel='epanechnikov')
            # kdeGrid = GridSearchCV(estimator=kde, param_grid=paramGrid, cv=3)
            # kde = kdeGrid.fit(x.reshape(-1, 1)).best_estimator_
            # logDens = kde.score_samples(x.reshape(-1, 1))
            # f = np.exp(logDens)

            p = dict()
            p["x"] = x
            p["f"] = f
            pdfs.append(p)
        probabilisticResults = {
            "pdfs": np.array(pdfs),
            "cdfs": np.array(cdfs)
        }
        self.dataset.validationP = probabilisticResults
        return probabilisticResults


class ForecastModelGRU(ForecastModelNNBase):
    """
    LSTM预报模型
    """

    def constructModel(self):
        optimizedHyperParameters = self.optimizedHyperParameters
        fixedHyperParameters = self.fixedHyperParameters

        hiddenLayerNum = optimizedHyperParameters["hiddenLayerNum"]
        hiddenLayerNodeNum = optimizedHyperParameters["hiddenLayerNodeNum"]
        dropoutRate = optimizedHyperParameters["dropoutRate"]
        regularizeRate = optimizedHyperParameters["regularizeRate"]

        lossNameInTraining = fixedHyperParameters["lossNameInTraining"]
        optimizer = fixedHyperParameters["optimizer"]

        regularize = regularizers.l1_l2(regularizeRate, regularizeRate)

        model = Sequential()
        for i in range(hiddenLayerNum):
            model.add(GRU(units=hiddenLayerNodeNum, recurrent_regularizer=regularize, activity_regularizer=regularize,
                          bias_regularizer=regularize, return_sequences=True))
            model.add(Dropout(dropoutRate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss=lossNameInTraining, optimizer=optimizer)
        self.model = model


class ForecastModelSWLSTM(ForecastModelNNBase):
    """
    LSTM预报模型
    """

    def constructModel(self):
        optimizedHyperParameters = self.optimizedHyperParameters
        fixedHyperParameters = self.fixedHyperParameters

        hiddenLayerNum = optimizedHyperParameters["hiddenLayerNum"]
        hiddenLayerNodeNum = optimizedHyperParameters["hiddenLayerNodeNum"]
        dropoutRate = optimizedHyperParameters["dropoutRate"]
        regularizeRate = optimizedHyperParameters["regularizeRate"]

        lossNameInTraining = fixedHyperParameters["lossNameInTraining"]
        optimizer = fixedHyperParameters["optimizer"]

        regularize = regularizers.l1_l2(regularizeRate, regularizeRate)

        model = Sequential()
        for i in range(hiddenLayerNum):
            model.add(SWLSTM(units=hiddenLayerNodeNum, recurrent_regularizer=regularize, activity_regularizer=regularize,
                           bias_regularizer=regularize, return_sequences=True))
        model.add(Dropout(dropoutRate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss=lossNameInTraining, optimizer=optimizer)
        self.model = model


class ForecastModelCNN(ForecastModelNNBase):
    """
    CNN预报模型
    """
    def constructModel(self):
        optimizedHyperParameters = self.optimizedHyperParameters
        fixedHyperParameters = self.fixedHyperParameters

        hiddenLayerNum = optimizedHyperParameters["hiddenLayerNum"]
        hiddenLayerNodeNum = optimizedHyperParameters["hiddenLayerNodeNum"]
        dropoutRate = optimizedHyperParameters["dropoutRate"]
        regularizeRate = optimizedHyperParameters["regularizeRate"]

        lossNameInTraining = fixedHyperParameters["lossNameInTraining"]
        optimizer = fixedHyperParameters["optimizer"]

        regularize = regularizers.l1_l2(regularizeRate, regularizeRate)

        model = Sequential()
        for i in range(hiddenLayerNum):
            model.add(Conv1D(filters=hiddenLayerNodeNum, kernel_size=1, strides=1, activation="relu",
                             kernel_regularizer=regularize, bias_regularizer=regularize,
                             activity_regularizer=regularize))
            model.add(MaxPooling1D(pool_size=1))
        model.add(Dropout(dropoutRate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss=lossNameInTraining, optimizer=optimizer)
        self.model = model


class ForecastModelANN(object):
    """
    LSTM预报模型
    """

    def constructModel(self):
        optimizedHyperParameters = self.optimizedHyperParameters
        fixedHyperParameters = self.fixedHyperParameters

        hiddenLayerNum = optimizedHyperParameters["hiddenLayerNum"]
        hiddenLayerNodeNum = optimizedHyperParameters["hiddenLayerNodeNum"]
        dropoutRate = optimizedHyperParameters["dropoutRate"]
        regularizeRate = optimizedHyperParameters["regularizeRate"]

        lossNameInTraining = fixedHyperParameters["lossNameInTraining"]
        optimizer = fixedHyperParameters["optimizer"]

        regularize = regularizers.l1_l2(regularizeRate, regularizeRate)

        model = Sequential()
        for i in range(hiddenLayerNum):
            model.add(LSTM(units=hiddenLayerNodeNum, recurrent_regularizer=regularize, activity_regularizer=regularize,
                           bias_regularizer=regularize, return_sequences=True))
            model.add(Dropout(dropoutRate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss=lossNameInTraining, optimizer=optimizer)
        self.model = model


class ForecastModelSVR(object):
    pass
