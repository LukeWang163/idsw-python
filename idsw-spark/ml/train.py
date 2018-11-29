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
        Spark version for training model
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
        self.logger.debug("using PySpark")
        from pyspark.ml import Pipeline
        self.logger.info("initializing SparkSession")
        self.spark = utils.init_spark()
        self.originalDF = self.dataUtil.SparkReadHive(self.inputUrl2, self.spark)
        self.model = Pipeline.load(self.inputUrl1).getStages()[0]

    def execute(self):
        featureCols = self.param["features"]
        labelCol = self.param["label"]

        # 训练Spark等模型
        from pyspark.ml import Pipeline
        import pyspark.ml.clustering
        from pyspark.ml.feature import VectorAssembler

        if not isinstance(self.model, pyspark.ml.clustering.KMeans):
            # 使用VectorAssembler将特征列聚合成一个DenseVector
            self.logger.info("Training model")
            vectorAssembler = VectorAssembler(inputCols=featureCols, outputCol="features")
            self.model.setParams(featuresCol="features", labelCol=labelCol)
            pipeline = Pipeline(stages=[vectorAssembler, self.model])
            self.pipelinemodel = pipeline.fit(self.originalDF)

    def setOut(self):
        self.logger.info("saving trained distributed model to %s" % self.outputUrl1)
        self.pipelinemodel.write().overwrite().save(self.outputUrl1)


class TrainClustering:
    def __init__(self, args, args2):
        """
        Spark version for training clustering model
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
        # 训练Spark等模型
        self.logger.debug("initializing SparkSession")
        from pyspark.ml import Pipeline

        self.spark = utils.init_spark()
        self.originalDF = self.dataUtil.SparkReadHive(self.inputUrl2, self.spark)
        self.model = Pipeline.load(self.inputUrl1).getStages()[0]

    def execute(self):
        featureCols = self.param["features"]

        # 训练Spark等模型
        from pyspark.ml import Pipeline
        import pyspark.ml.clustering
        from pyspark.ml.feature import VectorAssembler
        # 使用VectorAssembler将特征列聚合成一个DenseVector
        self.logger.info("training distributed clustering model")
        vectorAssembler = VectorAssembler(inputCols=featureCols, outputCol="features")
        self.model.setParams(featuresCol="features")
        pipeline = Pipeline(stages=[vectorAssembler, self.model])
        self.pipelinemodel = pipeline.fit(self.originalDF)
        self.transformDF = self.pipelinemodel.transform(self.originalDF).drop("features")

    def setOut(self):
        self.logger.info("saving trained distributed clustering model to %s" % self.outputUrl1)
        self.pipelinemodel.write().overwrite().save(self.outputUrl1)
        utils.dataUtil.SparkWriteHive(self.transformDF, self.outputUrl2)


class TuneHyperparameter:
    pass
