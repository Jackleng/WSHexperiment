from case.Case import CaseProcess
from dataset.Dataset import SmallDataOperateEnum, SmallDataFactory, Dataset
from forecast.Feature import FeatureGenerationOperateEnum
from util.const.GlobalConstants import GlobalConstants
from util.io.XmlUtils import XmlUtils
from util.io.InputOuputUtils import JsonInputOutputUtils, PandasInputOutputUtils

import numpy as np

rootPath = GlobalConstants.getRootPath()


def testXmlToDicAndJson():
    partPath = 'resources/test/caseTest.xml'
    path = rootPath + partPath
    # dict = XmlUtils.parseXmlToDict(path)
    js = XmlUtils.parseXmlToJson(path)

    partSavePath = 'resources/test/caseTest.json'
    savePath = rootPath + partSavePath
    JsonInputOutputUtils.saveJson(js, savePath)


def testDataset():
    partPath = 'resources/test/1518.xlsx'
    path = rootPath + partPath
    sheetName = "Sheet1"
    pandasData = PandasInputOutputUtils.readExcelDataToPandas(path, sheetName)
    numpyData = np.array(pandasData)


    featureIndexes = np.array(range(2, 10))
    operateEnums = np.array([[FeatureGenerationOperateEnum.Linear, FeatureGenerationOperateEnum.Power] for _ in range(2, 10)])
    transformFuncs = np.array([["y=1*x+0", "y=1*x^2+0"] for _ in range(2, 10)])
    historyIndexes = np.array([[i for i in range(3)], [i for i in range(1, 3)], [i for i in range(3)], [i for i in range(1, 3)],
                      [i for i in range(1, 3)], [i for i in range(1, 3)], [i for i in range(1, 3)],
                      [i for i in range(1, 3)], [i for i in range(1, 3)]])
    # historyIndexes = np.array([[j for j in range(3)] for _ in range(2, 10)])
    labelIndex = 5


    print("断点")

def testCase():
    partPath = 'resources/test/caseTest.xml'
    partPath = 'resources/test/lc.xml'
    path = rootPath + partPath

    dict = XmlUtils.parseXmlToDict(path)

    caseParams = dict["caseParams"]
    caseProcess = CaseProcess(caseParams)


if __name__ == "__main__":
    # 测试XML数据转dic和json
    # testXmlToDicAndJson()

    # 测试数据集读取
    # testDataset()

    testCase()
    print("TestMain-->__main__()中断点1")
