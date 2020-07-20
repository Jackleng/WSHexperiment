# -*- coding: utf-8 -*-
# @File         : XmlUtils.py
# @Author       : Zhendong Zhang
# @Organization : Huazhong University of Science and Technology
# @Email        : zzd_zzd@hust.edu.cn
# @Time         : 2020/6/28 15:01 

import xml.dom.minidom as xdm
import json

from util.const.GlobalConstants import GlobalEnumConstants


class XmlUtils(object):

    @staticmethod
    def parseXmlToDict(xmlPath):
        xmlDom = xdm.parse(xmlPath)
        root = xmlDom.documentElement
        key = root.nodeName

        xmlUtilsObject = XmlUtils()
        dic = dict()
        xmlUtilsObject.__recursiveConvert(root, dic, key)
        return dic

    @staticmethod
    def parseXmlToJson(xmlPath):
        dic = XmlUtils.parseXmlToDict(xmlPath)
        js = json.dumps(dic)
        return js

    def __recursiveConvert(self, root, dic, key, type=None):
        """
        递归将xml中的树转换成dict类型
        :param root: xml树
        :param dic: 转换后dict值
        :param key: 父节点的键
        :return: None
        """
        childNodes = root.childNodes
        childDic = dict()
        childList = []
        for childNode in childNodes:
            nodeName = childNode.nodeName
            if len(childNode.childNodes) == 0:
                nodeValue = childNode.nodeValue
                nodeValueRep = nodeValue.replace(" ", "").replace("\n", "")
                if (nodeName == "#text") & (nodeValueRep != ""):
                    if nodeValue == "None":
                        nodeValue = None
                    else:
                        if type is None:
                            pass
                        elif type == "String":
                            pass
                        elif type == "Integer":
                            nodeValue = int(float(nodeValue))
                        elif type == "Float":
                            nodeValue = float(nodeValue)
                        elif type == "Double":
                            nodeValue = float(nodeValue)
                        elif type == "Boolean":
                            nodeValue = (nodeValue == "True")
                        elif type == "Code":
                            # 其实基本所有类型的数据格式，只要严格按照python的语法来写都可以采用下面这套代码统一实现
                            code = nodeValue
                            localVariable = dict()
                            exec(code, localVariable)
                            nodeValue = localVariable['nodeValue']
                        elif type.startswith("Enum."):
                            nodeValue = GlobalEnumConstants.getEnum(type[5:], nodeValue)
                    dic[key] = nodeValue
            else:
                childType = childNode.getAttribute("type")
                if type == "MultiList":
                    childDi = dict()
                    self.__recursiveConvert(childNode, childDi, nodeName, childType)
                    childList.append(childDi)
                    dic[key] = childList
                else:
                    self.__recursiveConvert(childNode, childDic, nodeName, childType)
                    dic[key] = childDic
