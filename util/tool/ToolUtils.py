# -*- coding: utf-8 -*-
# @File         : ToolUtils.py
# @Author       : Zhendong Zhang
# @Organization : Huazhong University of Science and Technology
# @Email        : zzd_zzd@hust.edu.cn
# @Time         : 2020/6/29 16:18 

import numpy as np
import pickle
import os


class ToolNormalUtils(object):
    # 持久化变量
    @staticmethod
    def pickleData(path, data):
        dirPath = os.path.dirname(path)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        with open(path, 'wb') as file:
            pickle.dump(data, file);

    # 加载持久化的变量
    @staticmethod
    def loadData(path):
        if os.path.exists(path):
            with open(path, 'rb') as file:
                data = pickle.load(file)
        else:
            data = []
        return data

    @staticmethod
    def deleteFile(path):
        if os.path.exists(path):
            os.remove(path)


class NumpyUtils(object):

    @staticmethod
    def hStackByCutTail(a, b):
        """
        水平向拼接两个numpy数组，当a和b行数不一样时取短，长的那个断尾
        :param a: 数组1，二维数组
        :param b: 数组2，二维数组
        :return: 水平拼接后的数组，二维数组
        """

        aRowNum = a.shape[0]
        bRowNum = b.shape[0]
        if aRowNum > bRowNum:
            a = a[0:bRowNum, :]
        elif aRowNum < bRowNum:
            b = b[0:aRowNum, :]

        return np.hstack((a, b))

    @staticmethod
    def hStackByCutHead(a, b):
        """
        水平向拼接两个numpy数组，当a和b行数不一样时取短，长的那个去头
        :param a: 数组1，二维数组
        :param b: 数组2，二维数组
        :return: 水平拼接后的数组，二维数组
        """

        aRowNum = a.shape[0]
        bRowNum = b.shape[0]
        if aRowNum > bRowNum:
            a = a[aRowNum - bRowNum::, :]
        elif aRowNum < bRowNum:
            b = b[bRowNum - aRowNum::, :]

        return np.hstack((a, b))
