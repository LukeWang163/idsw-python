#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.ml.train.py
# @Desc    : Scripts for initializing binary classification models. 机器学习->模型训练
from .. import utils


class PyTrainModel:
    def __init__(self, args, args2):
        """
        Standalone and Spark version for training model
        @param args: dict
        featureCols: list
        labelCol: String
        """
        self.originalDF = None
        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]
        self.model = None
        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        if self.inputUrl1.endswith(".pkl"):
            # 训练sklearn等模型
            print("using scikit-learn")

            from sklearn.externals import joblib
            # self.originalDF = data.PyReadCSV(self.inputUrl2)
            self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
            self.model = joblib.load(self.inputUrl1)
        else:
            # 训练Spark等模型
            print("using PySpark")
            from pyspark.ml import Pipeline

            self.spark = utils.init_spark()
            self.originalDF = self.dataUtil.SparkReadHive(self.inputUrl2, self.spark)
            self.model = Pipeline.load(self.inputUrl1).getStages()[0]

    def execute(self):
        featureCols = self.param["features"]
        labelCol = self.param["label"]

        if self.inputUrl1.endswith(".pkl"):
            # 训练sklearn等模型
            import sklearn.cluster
            if (not isinstance(self.model, sklearn.cluster.k_means_.KMeans)) & (
                    not isinstance(self.model, sklearn.cluster.dbscan_.DBSCAN)):
                self.model.fit(self.originalDF[featureCols], self.originalDF[labelCol])
            else:
                self.model.fit(self.originalDF[featureCols])
        else:
            # 训练Spark等模型
            print("using PySpark")
            from pyspark.ml import Pipeline
            import pyspark.ml.clustering
            from pyspark.ml.feature import VectorAssembler
            if not isinstance(self.model, pyspark.ml.clustering.KMeans):
                # 使用VectorAssembler将特征列聚合成一个DenseVector
                vectorAssembler = VectorAssembler(inputCols=featureCols, outputCol="features")
                self.model.setParams(featuresCol="features", labelCol=labelCol)
                pipeline = Pipeline(stages=[vectorAssembler, self.model])
                self.pipelinemodel = pipeline.fit(self.originalDF)

    def setOut(self):
        if self.inputUrl1.endswith(".pkl"):
            from sklearn.externals import joblib
            joblib.dump(self.model, self.outputUrl1, compress=True)

        else:
            self.pipelinemodel.write().overwrite().save(self.outputUrl1)


class PyTrainClustering:
    pass


class PyTuneHyperparameter:
    pass
