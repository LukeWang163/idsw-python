#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.featuring.transform.py
# @Desc    : Scripts for feature transformation. 特征工程->特征变换
from .. import utils


class PyPCA:
    def __init__(self, args, args2):
        """
        Python version for conducting PCA on input dataset
        @param args: dict
        inputUrl1: String
        outputUrl1: String
        columns: list
        k: int
        """
        self.originalDF = None
        self.transformDF = None
        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]

        self.columns = args["param"]["columns"]
        self.k = args["param"]["k"]

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl1)

    def execute(self):
        from sklearn.decomposition import PCA
        import pandas as pd
        assert int(self.k) <= self.originalDF.shape[0], "维度需不大于样本数"
        self.transformDF = self.originalDF
        pca = PCA(n_components=self.k).fit_transform(self.originalDF[self.originalDF])
        pca_feature_names = ["PCA_" + str(i) for i in range(self.k)]
        for i, col in enumerate(pca_feature_names):
            self.transformDF[col] = pd.SparseSeries(pca[:, i], fill_value=0)

    def setOut(self):
        self.dataUtil.PyWriteHive(self.transformDF, self.outputUrl1)


class PyFLDA:
    pass


class PySVD:
    pass


class PyFeatureScale:
    pass


class PyFeatureSoften:
    pass


class PyAnomalyDetection:
    pass


class PyFeatureDiscrete:
    pass


class PyOneHot:
    pass


class SparkPCA:
    def __init__(self, args, args2):
        """
        Spark version for conducting PCA on input dataset
        @param args: dict
        inputUrl: String
        outputUrl: String
        columns: list
        k: int
        """
        self.originalDF = None
        self.transformDF = None
        self.pipelinemodel = None

        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]

        self.columns = args["param"]["columns"]
        self.k = args["param"]["k"]

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
        self.pipelinemodel = pipeline.fit(self.originalDF)

        self.transformDF = self.pipelinemodel.transform(self.originalDF)

        # 定义UDF，将vector转换成double array
        def to_array(col):
            def to_array_(v):
                return v.toArray().tolist()

            return udf(to_array_, ArrayType(DoubleType()))(col)

        self.transformDF = self.transformDF.withColumn("pca_features", to_array(col("pca_features")))

        for i in range(5):
            self.transformDF = self.transformDF.withColumn("PCA_" + str(i), col("pca_features")[i])
        self.transformDF = self.transformDF.drop("pca_features", "features")

    def setOut(self):
        utils.dataUtil.SparkWriteHive(self.transformDF, self.outputUrl1)


class SparkFLDA:
    pass


class SparkSVD:
    pass


class SparkFeatureScale:
    pass


class SparkFeatureSoften:
    pass


class SparkAnomalyDetection:
    pass


class SparkFeatureDiscrete:
    pass


class SparkOneHot:
    pass
