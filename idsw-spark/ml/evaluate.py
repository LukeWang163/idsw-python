#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.ml.evaluate.py
# @Desc    : Evaluation scripts for our built models.
import utils
import logging


class CrossValidate:
    pass


class EvaluateBinaryClassifier:
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
        self.param = args["param"]

        self.originalDF = None
        self.transformDF = None
        self.model = None
        self.result = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # Spark模型加载
        self.logger.debug("using PySpark")
        from pyspark.ml import PipelineModel

        self.logger.info("initializing SparkSession")
        self.spark = utils.init_spark()

        self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl2, self.spark)
        self.model = PipelineModel.load(self.inputUrl1)

    def execute(self):
        from collections import OrderedDict
        featureCols = self.param["features"]
        labelCol = self.param["label"]

        # Spark等模型评估
        from pyspark.sql.types import StructType, StructField, FloatType
        self.logger.info("evaluating model")
        self.transformDF = self.model.transform(self.originalDF).drop("features")
        # Spark评估二分类的接口不提供准确率等指标，因此先计算TP、TN、FN及FP，再根据公式计算准确率等评价指标
        TP = float(self.transformDF.filter(self.transformDF.prediction == 1.0)
                   .filter(self.transformDF[labelCol] == self.transformDF.prediction).count())
        TN = float(self.transformDF.filter(self.transformDF.prediction == 0.0)
                   .filter(self.transformDF[labelCol] == self.transformDF.prediction).count())
        FN = float(self.transformDF.filter(self.transformDF.prediction == 0.0)
                   .filter(~(self.transformDF[labelCol] == self.transformDF.prediction)).count())
        FP = float(self.transformDF.filter(self.transformDF.prediction == 1.0)
                   .filter(~(self.transformDF[labelCol] == self.transformDF.prediction)).count())

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall)
        self.logger.info("accuracy: %s" % accuracy)
        self.logger.info("precision: %s" % precision)
        self.logger.info("recall: %s" % recall)
        self.logger.info("f1 score: %s" % f1)

        result_struct = StructType(
            [StructField("accuracy", FloatType(), True), StructField("precision", FloatType(), True),
             StructField("recall", FloatType(), True), StructField("f1 score", FloatType(), True)])
        self.result = self.spark.createDataFrame(
            self.spark.sparkContext.parallelize([(accuracy, precision, recall, f1)]),
            result_struct)

    def setOut(self):
        self.logger.info("saving evaluation result to %s" % self.outputUrl1)
        utils.dataUtil.SparkWriteHive(self.result, self.outputUrl1)


class EvaluateMultiClassClassifier:
    def __init__(self, args, args2):
        """
        Spark version for evaluating multi-class classifier
        @param args: dict
        featureCols: list
        labelCol: string
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]

        self.originalDF = None
        self.transformDF = None
        self.model = None
        self.result = None
        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # Spark模型加载
        self.logger.debug("using PySpark")
        from pyspark.ml import PipelineModel
        self.logger.info("initializing SparkSession")
        self.spark = utils.init_spark()

        self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl2, self.spark)
        self.model = PipelineModel.load(self.inputUrl1)

    def execute(self):
        from collections import OrderedDict
        featureCols = self.param["features"]
        labelCol = self.param["label"]

        # Spark等模型评估
        from pyspark.sql.types import StructType, StructField, FloatType
        self.logger.info("evaluating model")
        self.transformDF = self.model.transform(self.originalDF).drop("features")
        # 与二分类评估不同的是，MulticlassClassificationEvaluator提供了准确率等评价指标的接口
        from pyspark.ml.evaluation import MulticlassClassificationEvaluator
        accuracy = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction",
                                                     metricName="accuracy").evaluate(self.transformDF)
        precision = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction",
                                                      metricName="weightedPrecision").evaluate(self.transformDF)
        recall = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction",
                                                   metricName="weightedRecall").evaluate(self.transformDF)
        f1 = MulticlassClassificationEvaluator(labelCol=labelCol, predictionCol="prediction",
                                               metricName="f1").evaluate(self.transformDF)

        self.logger.info("accuracy: %s" % accuracy)
        self.logger.info("precision: %s" % precision)
        self.logger.info("recall: %s" % recall)
        self.logger.info("f1 score: %s" % f1)

        result_struct = StructType(
            [StructField("accuracy", FloatType(), True), StructField("precision", FloatType(), True),
             StructField("recall", FloatType(), True), StructField("f1 score", FloatType(), True)])
        self.result = self.spark.createDataFrame(
            self.spark.sparkContext.parallelize([(accuracy, precision, recall, f1)]),
            result_struct)

    def setOut(self):
        self.logger.info("saving evaluation result to %s" % self.outputUrl1)
        utils.dataUtil.SparkWriteHive(self.result, self.outputUrl1)


class EvaluateRegressor:
    def __init__(self, args, args2):
        """
        Spark version for evaluating regressor
        @param args: dict
        featureCols: list
        labelCol: string
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]

        self.originalDF = None
        self.transformDF = None
        self.model = None
        self.result = None
        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # Spark等模型评估
        self.logger.debug("using PySpark")
        from pyspark.ml import PipelineModel
        self.logger.info("initializing SparkSession")
        self.spark = utils.init_spark()

        self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl2, self.spark)
        self.model = PipelineModel.load(self.inputUrl1)

    def execute(self):
        from collections import OrderedDict
        featureCols = self.param["features"]
        labelCol = self.param["label"]

        self.logger.info("using PySpark")
        from pyspark.ml.evaluation import RegressionEvaluator
        from pyspark.sql.types import StructType, StructField, FloatType

        self.logger.info("evaluating model")
        self.transformDF = self.model.transform(self.originalDF).drop("features")

        r2 = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="r2").evaluate(
            self.transformDF)
        rmse = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="rmse").evaluate(
            self.transformDF)
        mae = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="mae").evaluate(
            self.transformDF)

        self.logger.info("r2: %s" % r2)
        self.logger.info("rmse: %s" % rmse)
        self.logger.info("mae: %s" % mae)

        self.result_struct = StructType(
            [StructField("r2", FloatType(), True), StructField("rmse", FloatType(), True),
             StructField("mae", FloatType(), True)])
        self.result = self.spark.createDataFrame(self.spark.sparkContext.parallelize([(r2, rmse, mae)]),
                                                 self.result_struct)

    def setOut(self):
        self.logger.info("saving evaluation result to %s" % self.outputUrl1)
        utils.dataUtil.SparkWriteHive(self.result, self.outputUrl1)


class EvaluateClustering:
    def __init__(self, args, args2):
        """
        Spark version for evaluating clustering model
        @param args: dict
        featureCols: list
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]

        self.originalDF = None
        self.model = None
        self.result = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # Spark模型加载
        self.logger.debug("using PySpark")
        from pyspark.ml import PipelineModel
        self.logger.info("initialize SparkSession")
        self.spark = utils.init_spark()

        self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl2, self.spark)
        self.model = PipelineModel.load(self.inputUrl1)

    def execute(self):
        from collections import OrderedDict
        featureCols = self.param["features"]
        # Spark等模型评估
        self.logger.info("using PySpark")
        from pyspark.ml.evaluation import ClusteringEvaluator
        from pyspark.ml.feature import VectorAssembler
        from pyspark.sql.types import StructType, StructField, FloatType
        self.logger.info("evaluating model")
        transformDF = VectorAssembler(inputCols=featureCols, outputCol="features").transform(self.originalDF)

        silhouetteScore = ClusteringEvaluator().evaluate(transformDF)

        self.logger.info("silhouette: %s" % silhouetteScore)

        result_struct = StructType([StructField("silhouette", FloatType(), True)])

        self.result = self.spark.createDataFrame(self.spark.sparkContext.parallelize([silhouetteScore]),
                                                 result_struct)

    def setOut(self):
        self.logger.info("saving evaluation result to %s" % self.outputUrl1)
        utils.dataUtil.SparkWriteHive(self.result, self.outputUrl1)
