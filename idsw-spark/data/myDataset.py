#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 14:19
# @Author  : Luke
# @File    : myDataset.py
# @Desc    : read dataset from HDFS and then write to HIVE
import utils


class File2Hive:
    def __init__(self, args, args2):
        """
        Spark version for reading data from HDFS and then write to HIVE
        @param args: dict
        inputUrl: String
        outputUrl: String
        """
        self.outputUrl1 = args["output"][0]["value"]

        self.type = args["param"]["type"]
        self.inputUrl1 = args["param"]["path"]
        self.DF = None

        self.spark = utils.init_spark()

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        if self.type == "csv":
            self.DF = self.dataUtil.SparkReadCSV(self.inputUrl1, self.spark)

    def execute(self):
        return

    def setOut(self):
        self.dataUtil.SparkWriteHive(self.DF, self.outputUrl1)
