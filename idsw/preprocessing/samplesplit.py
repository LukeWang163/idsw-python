#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.preprocessing.samplesplit.py
# @Desc    : Scripts for sampling and spliting. 数据预处理->采样/过滤
import logging
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
        self.logger = logging.getLogger(self.__class__.__name__)
        self.originalDF = None
        self.DF1 = None
        self.DF2 = None
        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = args["output"][1]["value"]
        self.param = args["param"]
        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):

        # self.originalDF = data.PyReadCSV(self.inputUrl1)
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl1)

    def execute(self):

        def splitByRatio():
            # 根据设定的比例进行拆分
            from sklearn.model_selection import train_test_split
            ratio = float(self.param["ratio"])
            self.DF1, self.DF2 = train_test_split(self.originalDF, train_size=ratio)

        def splitByThreshold():
            # 根据对设定列设定的阈值进行拆分
            thresholdColumn = self.param["thresholdColumn"]
            threshold = float(self.param["threshold"])
            self.DF1 = self.originalDF[self.originalDF[thresholdColumn] >= threshold]
            self.DF2 = self.originalDF[self.originalDF[thresholdColumn] < threshold]

        mode = self.param["splitBy"]
        self.logger.info("splitting by %s" % mode)
        modes = {"byRatio": splitByRatio, "byThreshold": splitByThreshold}
        modes[mode]()

    def setOut(self):

        self.dataUtil.PyWriteHive(self.DF1, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.DF2, self.outputUrl2)
        # data.PyWriteCSV(self.DF1, self.outputUrl1)
        # data.PyWriteCSV(self.DF2, self.outputUrl2)
