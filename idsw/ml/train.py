#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.ml.train.py
# @Desc    : Scripts for initializing binary classification models. 机器学习->模型训练
import utils
import logging
import logging.config
logging.config.fileConfig('logging.ini')


class TrainModel:
    def __init__(self, args, args2):
        """
        Standalone version for training model
        @param args: dict
        featureCols: list
        labelCol: String
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.originalDF = None
        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]
        self.model = None
        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # 训练sklearn等模型
        self.logger.debug("using standalone model")
        # self.originalDF = data.PyReadCSV(self.inputUrl2)
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
        self.model = self.dataUtil.PyReadModel(self.inputUrl1)

    def execute(self):
        featureCols = self.param["features"]
        labelCol = self.param["label"]

        # 训练sklearn等模型
        import sklearn.cluster

        if (not isinstance(self.model, sklearn.cluster.k_means_.KMeans)) & (
                not isinstance(self.model, sklearn.cluster.dbscan_.DBSCAN)):
            self.logger.info("Training model")
            self.model.fit(self.originalDF[featureCols], self.originalDF[labelCol])

    def setOut(self):
        self.logger.info("saving trained standalone model to %s" % self.outputUrl1)
        self.dataUtil.PyWriteModel(self.model, self.outputUrl1)


class TrainClustering:
    def __init__(self, args, args2):
        """
        Standalone version for training clustering model
        @param args: dict
        featureCols: list
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.originalDF = None
        self.transformDF = None
        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = args["output"][1]["value"]
        self.param = args["param"]
        self.model = None
        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # 训练sklearn等模型
        self.logger.debug("using scikit-learn")

        # self.originalDF = data.PyReadCSV(self.inputUrl2)
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
        self.transformDF = self.originalDF.copy()
        self.model = self.dataUtil.PyReadModel(self.inputUrl1)

    def execute(self):
        featureCols = self.param["features"]

        # 训练sklearn等模型
        self.logger.info("training standalone clustering model")
        self.model.fit(self.originalDF[featureCols])
        self.transformDF["prediction"] = self.model.labels_

    def setOut(self):
        self.logger.info("saving trained standalone clustering model to %s" % self.outputUrl1)
        self.dataUtil.PyWriteModel(self.model, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.transformDF, self.outputUrl2)


class TuneHyperparameter:
    pass
