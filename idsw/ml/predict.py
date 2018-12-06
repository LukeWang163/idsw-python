#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/03
# @Author  : Luke
# @File    : idsw.ml.predict.py
# @Desc    : Scripts for generating predictions for test data based on trained models. 机器学习->预测->预测
import utils
import logging


class Predict:
    def __init__(self, args, args2):
        """
        Standalone version for predicting classification and regression algo
        @param args: dict
        featureCols: list
        labelCol: string
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.featureCols = args["param"]["features"]

        self.originalDF = None
        self.transformDF = None
        self.model = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # sklearn等模型加载
        self.logger.debug("using scikit-learn")
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
        # self.originalDF = data.PyReadCSV(self.inputUrl2)

        self.model = self.dataUtil.PyReadModel(self.inputUrl1)

    def execute(self):
        # sklearn等模型评估
        self.transformDF = self.originalDF.copy()
        self.logger.info(f"predicting {str(type(self.model))} model")
        # judge type
        modelType = None
        try:
            labelList = self.model.classes_
            self.logger.info("predicting classification model")

            if len(labelList) == 2:
                self.logger.info("predicting binary classification model")
                modelType = "binary"

            elif len(labelList) > 2:
                self.logger.info("predicting multi-class classification model")
                modelType = "multi"

        except AttributeError as e:
            import sklearn.cluster
            if (not isinstance(self.model, sklearn.cluster.k_means_.KMeans)) & (
                    not isinstance(self.model, sklearn.cluster.dbscan_.DBSCAN)):
                self.logger.info("predicting regression model")
                modelType = "reg"

            else:
                self.logger.error("not supported")
                import sys
                sys.exit(0)

        def executeBinary():
            predicted = self.model.predict(self.originalDF[self.featureCols])
            predicted_proba = self.model.predict_proba(self.originalDF[self.featureCols])
            self.transformDF["prediction"] = predicted
            for i in range(len(labelList)):
                self.transformDF["predicted as '%s'" % str(labelList[i])] = predicted_proba[:, i]

        def executeMulti():
            predicted = self.model.predict(self.originalDF[self.featureCols])
            predicted_proba = self.model.predict_proba(self.originalDF[self.featureCols])
            self.transformDF["prediction"] = predicted
            for i in range(len(labelList)):
                self.transformDF["predicted as '%s'" % str(labelList[i])] = predicted_proba[:, i]

        def executeReg():
            predicted = self.model.predict(self.originalDF[self.featureCols])
            self.transformDF["prediction"] = predicted

        # 三类模型
        modelTypes = {"binary": executeBinary, "multi": executeMulti, "reg": executeReg}
        if modelType is not None:
            modelTypes[modelType]()
        else:
            self.logger.error("not supported")
            import sys
            sys.exit(0)

    def setOut(self):
        self.logger.info("saving predicting result to %s" % self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.transformDF, self.outputUrl1)


class AssignToCluster:
    def __init__(self, args, args2):
        """
        Standalone version for predicting clutering algo
        @param args: dict
        featureCols: list
        labelCol: string
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.featureCols = args["param"]["features"]

        self.originalDF = None
        self.transformDF = None
        self.model = None
        self.result = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # sklearn等模型加载
        self.logger.debug("using scikit-learn")
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
        # self.originalDF = data.PyReadCSV(self.inputUrl2)

        self.model = self.dataUtil.PyReadModel(self.inputUrl1)

        import sklearn.cluster
        if (not isinstance(self.model, sklearn.cluster.k_means_.KMeans)) & (
                not isinstance(self.model, sklearn.cluster.dbscan_.DBSCAN)):
            self.logger.error("not supported")
            import sys
            sys.exit(0)

    def execute(self):
        # sklearn等模型评估
        self.transformDF = self.originalDF.copy()
        self.logger.info("predicting %s model" % (str(type(self.model))))
        predicted = self.model.predict(self.originalDF[self.featureCols])

        self.transformDF["prediction"] = predicted

    def setOut(self):
        self.logger.info("saving predicting result to %s" % self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.result, self.outputUrl1)
