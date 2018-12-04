#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/03
# @Author  : Luke
# @File    : idsw.ml.predict.py
# @Desc    : Scripts for generating predictions for test data based on trained models. 机器学习->预测->预测
import utils
import logging
import logging.config
logging.config.fileConfig('logging.ini')


class Predict:
    def __init__(self, args, args2):
        """
        Spark version for evaluating binary classifier
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

        self.logger.debug("initializing SparkSession")
        self.spark = utils.init_spark()

    def getIn(self):
        # Spark等模型加载
        self.logger.debug("using PySpark")

        self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl2, self.spark)

        from pyspark.ml import PipelineModel
        self.model = PipelineModel.load(self.inputUrl1)

    def execute(self):
        # Spark等模型预测
        self.logger.info("predicting %s model" % (str(type(self.model))))
        # judge type
        modelType = None
        try:
            if len(self.model.stages) == 4:
                labelList = self.model.stages[2].numClasses

            elif len(self.model.stages) == 2:
                labelList = self.model.stages[1].numClasses

            else:
                labelList = None
                self.logger.error("not supported pipelinemodel")

            self.logger.info("predicting classification model")

            if len(labelList) == 2:
                self.logger.info("predicting binary classification model")
                modelType = "binary"

            elif len(labelList) > 2:
                self.logger.info("predicting multi-class classification model")
                modelType = "multi"

        except AttributeError as e:
            import pyspark.ml.clustering
            if not isinstance(self.model, pyspark.ml.clustering.KMeans):
                self.logger.info("predicting regression model")
                modelType = "reg"
            else:
                self.logger.error("not supported")
                import sys
                sys.exit(0)

        def executeCla():
            from pyspark.sql.functions import udf, col
            from pyspark.sql.types import ArrayType, DoubleType

            def to_array(col):
                def to_array_(v):
                    return v.toArray().tolist()

                return udf(to_array_, ArrayType(DoubleType()))(col)

            if len(self.model.stages) == 4:
                self.transformDF = self.model.transform(self.originalDF) \
                    .drop("features", "indexedLabel", "rawPrediction", "prediction") \
                    .withColumnRenamed("originalLabel", "prediction") \
                    .withColumn("predicted_proba", to_array(col("probability")))

                for i in range(len(labelList)):
                    self.transformDF = self.transformDF\
                        .withColumn("predicted as '%s'" % str(labelList[i]), col("predicted_proba")[i])
                self.transformDF = self.transformDF.drop("probability", "predicted_proba")

            elif len(self.model.stages) == 2:
                self.transformDF = self.model.transform(self.originalDF) \
                    .drop("features", "rawPrediction") \
                    .withColumn("predicted_proba", to_array(col("probability")))

                for i in range(len(labelList)):
                    self.transformDF = self.transformDF \
                        .withColumn("predicted as '%s'" % str(labelList[i]), col("predicted_proba")[i])
                self.transformDF = self.transformDF.drop("probability", "predicted_proba")

        def executeReg():
            self.transformDF = self.model.transform(self.originalDF) \
                .drop("features")

        # 三类模型
        modelTypes = {"binary": executeCla, "multi": executeCla, "reg": executeReg}
        if modelType is not None:
            modelTypes[modelType]()
        else:
            self.logger.error("not supported")
            import sys
            sys.exit(0)

    def setOut(self):
        self.logger.info("saving prediction result to %s" % self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        utils.dataUtil.SparkWriteHive(self.transformDF, self.outputUrl1)


class AssignToCluster:
    def __init__(self, args, args2):
        """
        Spark version for predicting clutering algo
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

        self.logger.debug("initializing SparkSession")
        self.spark = utils.init_spark()

    def getIn(self):
        # Spark等模型加载
        self.logger.debug("using PySpark")

        self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl2, self.spark)

        from pyspark.ml import PipelineModel
        self.model = PipelineModel.load(self.inputUrl1)

        import pyspark.ml.clustering
        if not isinstance(self.model, pyspark.ml.clustering.KMeans):
            self.logger.error("not supported")
            import sys
            sys.exit(0)

    def execute(self):
        # Spark等模型预测
        self.logger.info("predicting %s model" % (str(type(self.model))))
        self.transformDF = self.model.transform(self.originalDF).drop("features")

    def setOut(self):
        self.logger.info("saving predicting result to %s" % self.outputUrl1)
        # data.SparkWriteCSV(self.result, self.outputUrl1)
        utils.dataUtil.SparkWriteHive(self.result, self.outputUrl1)

