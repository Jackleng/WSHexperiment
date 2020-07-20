# -*- coding: utf-8 -*-
# @File         : Draw.py
# @Author       : Zhendong Zhang
# @Email        : zzd_zzd@hust.edu.cn
# @University   : Huazhong University of Science and Technology
# @Date         : 2019/8/14
# @Software     : PyCharm
# -*---------------------------------------------------------*-
import math

import matplotlib.pyplot as plt
from statsmodels.graphics.api import qqplot
import numpy as np
import os
from scipy import stats

plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False


class Draw(object):

    @staticmethod
    def drawPredictions(predictions, observations, lower=None, upper=None, alpha=None,
                        xlabel="时段", ylabel="", title="", isShow=True, isSave=False, savePath=None):
        lw = 2
        fontsize = 20
        index = [i for i in range(observations.shape[0])]

        fig, axs = plt.subplots(1, 1, figsize=(12, 6))

        axs.plot(index, observations, marker="o", markersize=3, label="observations", lw=lw, color='r')
        preColor = (0.11765, 0.56471, 1)
        axs.plot(index, predictions, marker="D", markersize=3, label="predictions", lw=lw, color=preColor)
        if lower is not None and upper is not None and alpha is not None:
            axs.plot(index, lower, color=preColor, lw=lw / 2)
            axs.plot(index, upper, color=preColor, lw=lw / 2)
            axs.fill_between(index, lower, upper, alpha=1, color=(0.9, 0.9, 0.9),
                             label='' + alpha + ' prediction interval')
            for i in range(len(index)):
                if observations[i] > upper[i] or observations[i] < lower[i]:
                    axs.plot(index[i], observations[i], marker="o", markersize=3, color='g')

        residual = observations - predictions
        axs2 = axs.twinx()
        axs2.bar(index, residual, fc='grey', label='residual error')
        axs2.set_ylabel("残差", fontsize=fontsize)
        axs2.set_ylim([round(min(residual) * 11), math.ceil(max(residual) * 1.6)])

        axs.grid()

        axs.set_title(title, loc="left", fontsize=fontsize)
        axs.set_xlim([min(index), max(index)])
        axs.set_xlabel(xlabel, fontsize=fontsize)
        axs.set_ylabel(ylabel, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)

        fig.legend(fontsize=fontsize - 5, bbox_transform=axs.transAxes)
        plt.tight_layout()
        if isShow:
            plt.show()
        dirPath = os.path.dirname(savePath)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        if isSave:
            fig.savefig(savePath, bbox_inches="tight", dpi=300)
        plt.close()

    @staticmethod
    def drawPDForCDF(x, f, observation, xlabel, ylabel, title, isShow=False, isSave=False, savePath=None):
        lw = 4
        fontsize = 24
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(x, f, '-', label="function", lw=lw)
        ax.plot(np.array([observation, observation]), np.array([0, max(f)]), 'r-', label='observation', lw=lw)
        ax.fill_between(x, np.zeros(len(f)), f, alpha=1, color=[0.9, 0.9, 0.9])
        xyRange = plt.axis()
        ax.set_ylim([0, xyRange[3] * 1.01])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        ax.text(observation, 0.02, str(format(observation, '.2f')), color='blueviolet', fontsize=fontsize,
                fontweight='bold')
        ax.set_title(title, loc="left", fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=10)
        plt.grid()
        if isShow:
            plt.show()
        dirPath = os.path.dirname(savePath)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        if isSave:
            fig.savefig(savePath, bbox_inches="tight", dpi=300)
        plt.close()

    @staticmethod
    def drawPIT(data, cdf=stats.uniform, xlabel="uniform distribution", ylabel="PIT", title="", isShow=False, isSave=False, savePath=None):
        lw = 4
        fontsize = 40
        fig = plt.figure(figsize=(16, 16))
        axs = fig.add_subplot(111)
        fig = qqplot(data, dist=cdf, line='45', ax=axs)
        deta = 1.358/(len(data))**0.5*(2**0.5)
        axs.plot([deta, 1], [0, 1-deta], '--', color='blueviolet', lw=lw, label='Kolmogorov 5% significance band')
        axs.plot([0, 1-deta], [deta, 1], '--', color='blueviolet', lw=lw)
        axs.set_title(title, loc="center", fontsize=fontsize)
        axs.set_xlabel(xlabel, fontsize=fontsize)
        axs.set_ylabel(ylabel, fontsize=fontsize)
        axs.set_xlim([0, 1])
        axs.set_ylim([0, 1])
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.grid()
        plt.legend(fontsize=25)
        if isShow:
            plt.show()
        dirPath = os.path.dirname(savePath)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        if isSave:
            fig.savefig(savePath, bbox_inches="tight", dpi=300)
        plt.close()
