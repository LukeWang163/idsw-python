#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 14:19
# @Author  : Luke
# @File    : myDataset.py
# @Desc    : read dataset from HDFS and then write to HIVE
import logging
import utils


class File2Hive:
    def __init__(self, args, args2):
        """
        Spark version for reading data from HDFS and then write to HIVE
        @param args: dict
        inputUrl: String
        outputUrl: String
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.originalDF = None
        self.outputUrl1 = args["output"][0]["value"]

        self.type = args["param"]["type"]
        self.inputUrl1 = args["param"]["path"]
        self.DF = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        if self.type == "csv":
            self.DF = self.dataUtil.PyReadCSV(self.inputUrl1)

    def execute(self):
        return

    def setOut(self):
        self.dataUtil.PyWriteHive(self.DF, self.outputUrl1)
