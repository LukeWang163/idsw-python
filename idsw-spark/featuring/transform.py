#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.featuring.transform.py
# @Desc    : Scripts for feature transformation. 特征工程->特征变换
import utils
import logging


class PCA:
    def __init__(self, args, args2):
        """
        Spark version for conducting PCA on input dataset
        @param args: dict
        inputUrl: String
        outputUrl: String
        columns: list
        k: int
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.originalDF = None
        self.transformDF = None
        self.pipelineModel = None

        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]

        self.columns = args["param"]["columns"]
        self.k = args["param"]["k"]

        self.logger.info("initializing SparkSession")
        self.spark = utils.init_spark()

    def getIn(self):
        self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl1, self.spark)

    def execute(self):
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.feature import PCA
        from pyspark.ml import Pipeline
        from pyspark.sql.functions import udf, col
        from pyspark.sql.types import ArrayType, DoubleType
        assert int(self.k) <= len(self.originalDF.columns), "维度需不大于样本数"

        # 需要将features合并到一个vector，稍后再分解
        vectorAssembler = VectorAssembler(inputCols=self.columns, outputCol="features")
        pca = PCA(k=5, inputCol="features", outputCol='pca_features')
        pipeline = Pipeline(stages=[vectorAssembler, pca])
        self.pipelineModel = pipeline.fit(self.originalDF)

        self.transformDF = self.pipelineModel.transform(self.originalDF)

        # 定义UDF，将vector转换成double array
        def to_array(col):
            def to_array_(v):
                return v.toArray().tolist()

            return udf(to_array_, ArrayType(DoubleType()))(col)

        self.transformDF = self.transformDF.withColumn("pca_features", to_array(col("pca_features")))

        for i in range(5):
            self.transformDF = self.transformDF.withColumn("pca_" + str(i), col("pca_features")[i])
        self.transformDF = self.transformDF.drop("pca_features", "features")

    def setOut(self):
        utils.dataUtil.SparkWriteHive(self.transformDF, self.outputUrl1)


class FLDA:
    pass


class SVD:
    pass


class FeatureScale:
    pass


class FeatureSoften:
    pass


class AnomalyDetection:
    pass


class FeatureDiscrete:
    pass


class OneHot:
    pass
