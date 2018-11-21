#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 14:19
# @Author  : Luke
# @File    : myDataset.py
# @Desc    : read dataset from HDFS and then write to HIVE

from .data import dataUtil
from .. import utils


class SparkFile2Hive:
    def __init__(self, args, args2):
        """
        Spark version for reading data from HDFS and then write to HIVE
        @param args: dict
        inputUrl: String
        outputUrl: String
        """
        self.originalDF = None
        self.outputUrl1 = args["output"][0]["value"]

        self.type = args["param"]["type"]
        self.inputUrl1 = args["param"]["path"]
        self.DF = None

        self.spark = utils.init_spark()

        self.dataUtil = dataUtil(args2)

    def getIn(self):
        if self.type == "csv":
            self.DF = self.dataUtil.SparkReadCSV(self.inputUrl1, self.spark)

    def execute(self):
        return

    def setOut(self):
        self.dataUtil.SparkWriteHive(self.DF, self.outputUrl1)
