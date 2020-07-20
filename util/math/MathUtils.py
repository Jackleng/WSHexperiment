# -*- coding: utf-8 -*-
# @File         : MathUtils.py
# @Author       : Zhendong Zhang
# @Organization : Huazhong University of Science and Technology
# @Email        : zzd_zzd@hust.edu.cn
# @Time         : 2020/6/29 15:17 
from scipy import interpolate
import numpy as np


class MathNormalUtils(object):
    """
    不做具体分类的数学工具类
    """

    @staticmethod
    def toBinaryWithFixedLength(num, length):
        return MathNormalUtils.toBinary(num).rjust(length, "0")

    @staticmethod
    def toBinary(num):
        if num > 0:
            return MathNormalUtils.toBinary(num // 2) + str(num % 2)
        return ""


class MathTransformUtils(object):
    """
    数学变换工具类
    """

    @staticmethod
    def linearTransformFunc(data, transformFunc):
        """
        线性变换函数
        :param data: 变换前数据
        :param transformFunc: 线性表达式，必须严格为y=a*x+b形式，大小写、中英文、多余空格等字符都不能错
        :return: 变换后数据
        """

        ss = transformFunc
        try:
            aStr = ss[ss.find("=") + 1:ss.find("*")]
            bStr = ss[ss.find("+") + 1::]
            a = float(aStr)
            b = float(bStr)
        except Exception as e:
            print(e)
            raise ValueError("TransformMathUtils-->linearTransformFunc("
                             ")方法中transformFunc参数格式错误，必须严格为y=a*x+b形式，大小写、中英文、多余空格等字符都不能错")

        newData = a * data + b
        return newData

    @staticmethod
    def powerTransformFunc(data, transformFunc):
        """
        幂变换函数
        :param data: 变换前数据
        :param transformFunc: 幂表达式，必须严格为y=a*x^b+c形式，大小写、中英文、多余空格等字符都不能错
        :return: 变换后数据
        """

        ss = transformFunc
        try:
            aStr = ss[ss.find("=") + 1:ss.find("*")]
            bStr = ss[ss.find("^") + 1:ss.find("+")]
            cStr = ss[ss.find("+") + 1::]
            a = float(aStr)
            b = float(bStr)
            c = float(cStr)
        except Exception as e:
            print(e)
            raise ValueError("TransformMathUtils-->powerTransformFunc("
                             ")方法中transformFunc参数格式错误，必须严格为y=a*x^b+c形式，大小写、中英文、多余空格等字符都不能错")

        newData = a * pow(data, b) + c
        return newData


class MathInterpolateUtils(object):
    @staticmethod
    def interp1d(X, Y, x, kind="cubic"):
        """
        一维插值
        :param X: 一维数组
        :param Y: 一维数组
        :param x: 一个数或者一个一维数组
        :param kind: 插值方法 nearest, zero为阶梯插值; slinear 线性插值; quadratic, cubic 为2阶、3阶B样条曲线插值
        :return: 插值结果
        """
        f = interpolate.interp1d(X, Y, kind=kind)
        xMax = max(X)
        xMin = min(X)
        if type(x) is np.ndarray:
            for i in range(len(x)):
                if x[i] < xMin:
                    x[i] = xMin
                if x[i] > xMax:
                    x[i] = xMax
        else:
            if x < xMin:
                x = xMin
            if x > xMax:
                x = xMax
        y = f(x)
        return y
