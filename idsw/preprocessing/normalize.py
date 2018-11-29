#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.preprocessing.normalize.py
# @Desc    : Scripts for normalizing data in different ways. 数据预处理->数据标准化
import utils


class GroupIntoBins:
    pass


class NormalizeData:
    def __init__(self, args, args2):
        """
        Standalone version for normalizing data, including minmax and zscore
        @param args: dict
        columns: list
        """
        self.originalDF = None
        self.transformedDF = None
        self.parameterDF = None
        self.inputUrl1 = args["input"][0]["value"]
        try:
            self.inputUrl2 = args["input"][1]["value"]
        except IndexError:
            self.inputUrl2 = ""
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = args["output"][1]["value"]
        try:
            self.columns = args["param"]["columns"]
        except KeyError as e:
            self.columns = None
        self.param = args["param"]
        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # self.originalDF = data.PyReadCSV(self.inputUrl1)
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl1)
        if self.outputUrl2 == "":
            print("will execute fit_transform")
        else:
            # self.parameterDF = data.PyReadCSV(self.inputUrl2)
            self.parameterDF = self.dataUtil.PyReadHive(self.inputUrl2)

    def execute(self):
        import pandas as pd

        def minMax():
            if (self.parameterDF is None) & (self.columns is not None):
                # 没有提供参数表，执行fit_transform操作
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                self.transformedDF = self.originalDF.copy()
                self.transformedDF[self.columns] = scaler.fit_transform(self.originalDF[self.columns])
                self.parameterDF = pd.DataFrame([scaler.data_min_, scaler.data_max_], columns=self.columns)
            elif self.parameterDF is not None:
                self.transformedDF = self.originalDF.copy()
                # 读取参数表中的参数并用于转换原始表
                for col in self.parameterDF.columns:
                    p_min = self.parameterDF.loc[0, col]
                    p_max = self.parameterDF.loc[1, col]
                    self.transformedDF[col] = (self.transformedDF[col] - p_min) / (p_max - p_min)

        def standard():
            if (self.parameterDF is None) & (self.columns is not None):
                # 没有提供参数表，执行fit_transform操作
                from sklearn.preprocessing import StandardScaler
                standardScaler = StandardScaler()
                self.transformedDF = self.originalDF.copy()
                self.transformedDF[self.columns] = standardScaler.fit_transform(self.originalDF[self.columns])
                self.parameterDF = pd.DataFrame([standardScaler.mean_, standardScaler.var_],
                                                columns=self.columns)

            elif self.parameterDF is not None:
                # 读取参数表中的参数并用于转换原始表
                self.transformedDF = self.originalDF.copy()
                for col in self.parameterDF.columns:
                    p_mean = self.parameterDF.loc[0, col]
                    p_std = self.parameterDF.loc[1, col]
                    self.transformedDF[col] = (self.transformedDF[col] - p_mean) / p_std

        # 归一化还是标准化
        mode = self.param["mode"]
        modes = {"minMax": minMax, "standard": standard}
        modes[mode]()

    def setOut(self):
        self.dataUtil.PyWriteCSV(self.transformedDF, self.outputUrl1)
        self.dataUtil.PyWriteCSV(self.parameterDF, self.outputUrl2)
