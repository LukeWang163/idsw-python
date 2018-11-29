#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.preprocessing.samplesplit.py
# @Desc    : Scripts for sampling and spliting. 数据预处理->采样/过滤
import utils


class SampleData:
    pass


class SplitData:

    def __init__(self, args, args2):
        """
         Standalone version for split data, including byRatio and byThreshold
         @param args: dict
         splitBy: string one of byRation and byThreshold
         ratio: double
         thresholdColumn: stirng
         threshold: double
         """
        self.originalDF = None
        self.DF1 = None
        self.DF2 = None

        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = args["output"][1]["value"]

        self.param = args["param"]

        print("using PySpark")

        self.spark = utils.init_spark()
        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):

        self.originalDF = self.dataUtil.SparkReadHive(self.inputUrl1, self.spark)

    def execute(self):

        def splitByRatio():
            # 根据设定的比例进行拆分
            ratio = float(self.param["ratio"])
            self.DF1, self.DF2 = self.originalDF.randomSplit([ratio, 1-ratio])

        def splitByThreshold():
            # 根据对设定列设定的阈值进行拆分
            thresholdColumn = self.param["thresholdColumn"]
            threshold = float(self.param["threshold"])
            self.DF1 = self.originalDF.filter(self.originalDF[thresholdColumn] >= threshold)
            self.DF2 = self.originalDF.filter(self.originalDF[thresholdColumn] < threshold)

        mode = self.param["splitBy"]
        modes = {"byRatio": splitByRatio, "byThreshold": splitByThreshold}
        modes[mode]()

    def setOut(self):
        self.dataUtil.SparkWriteHive(self.DF1, self.outputUrl1)
        self.dataUtil.SparkWriteHive(self.DF2, self.outputUrl2)
