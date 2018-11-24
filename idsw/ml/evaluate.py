#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.ml.evaluate.py
# @Desc    : Evaluation scripts for our built models.
from .. import utils


class PyCrossValidate:
    pass


class PyEvaluateBinaryClassifier:
    def __init__(self, args, args2):
        """
        Standalone and Spark version for evaluating binary classifier
        @param args: dict
        featureCols: list
        labelCol: string
        """
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
        if self.inputUrl1.endswith(".pkl"):
            # sklearn等模型加载
            print("using scikit-learn")
            import pandas as pd
            from sklearn.externals import joblib
            self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
            # self.originalDF = data.PyReadCSV(self.inputUrl2)

            self.model = joblib.load(self.inputUrl1)
        else:
            # Spark模型加载
            print("using PySpark")
            from pyspark.ml import PipelineModel

            self.spark = utils.init_spark()

            self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl2, self.spark)
            self.model = PipelineModel.load(self.inputUrl1)

    def execute(self):
        from collections import OrderedDict
        featureCols = self.param["features"]
        labelCol = self.param["label"]
        if self.inputUrl1.endswith(".pkl"):
            # sklearn等模型评估
            import pandas as pd
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            self.transformDF = self.originalDF.copy()
            predicted = self.model.predict(self.originalDF[featureCols])

            self.transformDF["prediction"] = predicted

            accuracy = accuracy_score(self.originalDF[labelCol], predicted)
            precision = precision_score(self.originalDF[labelCol], predicted)
            recall = recall_score(self.originalDF[labelCol], predicted)
            f1 = f1_score(self.originalDF[labelCol], predicted)

            self.result = pd.DataFrame.from_dict(
                OrderedDict({"accuracy": [accuracy], "precision": [precision], "recall": [recall], "f1 score": [f1]}))

        else:
            # Spark等模型评估
            print("using PySpark")
            from pyspark.sql.types import StructType, StructField, FloatType

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
            print("accuracy: %s" % accuracy)
            print("precision: %s" % precision)
            print("recall: %s" % recall)
            print("f1 score: %s" % f1)

            result_struct = StructType(
                [StructField("accuracy", FloatType(), True), StructField("precision", FloatType(), True),
                 StructField("recall", FloatType(), True), StructField("f1 score", FloatType(), True)])
            self.result = self.spark.createDataFrame(
                self.spark.sparkContext.parallelize([(accuracy, precision, recall, f1)]),
                result_struct)
            # output

    def setOut(self):
        if self.inputUrl1.endswith(".pkl"):
            # data.PyWriteCSV(self.result, self.outputUrl1)
            self.dataUtil.PyWriteHive(self.result, self.outputUrl1)

        else:
            utils.dataUtil.SparkWriteHive(self.result, self.outputUrl1)


class PyEvaluateMultiClassClassifier:
    def __init__(self, args, args2):
        """
        Standalone and Spark version for evaluating multi-class classifier
        @param args: dict
        featureCols: list
        labelCol: string
        """
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
        if self.inputUrl1.endswith(".pkl"):
            # sklearn等模型加载
            print("using scikit-learn")
            import pandas as pd
            from sklearn.externals import joblib
            self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
            # self.originalDF = data.PyReadCSV(self.inputUrl2)

            self.model = joblib.load(self.inputUrl1)
        else:
            # Spark模型加载
            print("using PySpark")
            from pyspark.ml import PipelineModel

            self.spark = utils.init_spark()

            self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl2, self.spark)
            self.model = PipelineModel.load(self.inputUrl1)

    def execute(self):
        from collections import OrderedDict
        featureCols = self.param["features"]
        labelCol = self.param["label"]
        if self.inputUrl1.endswith(".pkl"):
            # sklearn等模型评估
            import pandas as pd
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            self.transformDF = self.originalDF.copy()
            predicted = self.model.predict(self.originalDF[featureCols])

            self.transformDF["prediction"] = predicted

            accuracy = accuracy_score(self.originalDF[labelCol], predicted)
            precision = precision_score(self.originalDF[labelCol], predicted)
            recall = recall_score(self.originalDF[labelCol], predicted)
            f1 = f1_score(self.originalDF[labelCol], predicted)

            self.result = pd.DataFrame.from_dict(
                OrderedDict({"accuracy": [accuracy], "precision": [precision], "recall": [recall], "f1 score": [f1]}))

        else:
            # Spark等模型评估
            print("using PySpark")
            from pyspark.sql.types import StructType, StructField, FloatType

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

            print("accuracy: %s" % accuracy)
            print("precision: %s" % precision)
            print("recall: %s" % recall)
            print("f1 score: %s" % f1)

            result_struct = StructType(
                [StructField("accuracy", FloatType(), True), StructField("precision", FloatType(), True),
                 StructField("recall", FloatType(), True), StructField("f1 score", FloatType(), True)])
            self.result = self.spark.createDataFrame(
                self.spark.sparkContext.parallelize([(accuracy, precision, recall, f1)]),
                result_struct)
            # output

    def setOut(self):
        if self.inputUrl1.endswith(".pkl"):
            # data.PyWriteCSV(self.result, self.outputUrl1)
            self.dataUtil.PyWriteHive(self.result, self.outputUrl1)

        else:
            utils.dataUtil.SparkWriteHive(self.result, self.outputUrl1)


class PyEvaluateRegressor:
    def __init__(self, args, args2):
        """
        Standalone and Spark version for evaluating regressor
        @param args: dict
        featureCols: list
        labelCol: string
        """
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
        if self.inputUrl1.endswith(".pkl"):
            # sklearn等模型评估
            print("using scikit-learn")
            import pandas as pd
            from sklearn.externals import joblib
            self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
            # self.originalDF = utils.dataUtil.PyReadCSV(self.inputUrl2)
            self.model = joblib.load(self.inputUrl1)
        else:
            # Spark等模型评估
            print("using PySpark")
            from pyspark.ml import PipelineModel

            self.spark = utils.init_spark()

            self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl2, self.spark)
            self.model = PipelineModel.load(self.inputUrl1)

    def execute(self):
        from collections import OrderedDict
        featureCols = self.param["features"]
        labelCol = self.param["label"]

        if self.inputUrl1.endswith(".pkl"):
            import pandas as pd
            import numpy as np
            from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
            self.transformDF = self.originalDF.copy()
            predicted = self.model.predict(self.originalDF[featureCols])

            self.transformDF["prediction"] = predicted

            r2 = r2_score(self.originalDF[labelCol], predicted)
            rmse = np.sqrt(mean_squared_error(self.originalDF[labelCol], predicted))
            mae = median_absolute_error(self.originalDF[labelCol], predicted)

            self.result = pd.DataFrame.from_dict(
                OrderedDict({"r2": [r2], "rmse": [rmse], "mae": [mae]}))

        else:
            print("using PySpark")
            from pyspark.ml.evaluation import RegressionEvaluator
            from pyspark.sql.types import StructType, StructField, FloatType

            self.transformDF = self.model.transform(self.originalDF).drop("features")

            r2 = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="r2").evaluate(
                self.transformDF)
            rmse = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="rmse").evaluate(
                self.transformDF)
            mae = RegressionEvaluator(labelCol=labelCol, predictionCol="prediction", metricName="mae").evaluate(
                self.transformDF)

            print("r2: %s" % r2)
            print("rmse: %s" % rmse)
            print("mae: %s" % mae)

            self.result_struct = StructType(
                [StructField("r2", FloatType(), True), StructField("rmse", FloatType(), True),
                 StructField("mae", FloatType(), True)])
            self.result = self.spark.createDataFrame(self.spark.sparkContext.parallelize([(r2, rmse, mae)]),
                                                     self.result_struct)

    def setOut(self):
        if self.inputUrl1.endswith(".pkl"):
            # data.PyWriteCSV(self.result, self.outputUrl1)
            self.dataUtil.PyWriteHive(self.result, self.outputUrl1)

        else:
            utils.dataUtil.SparkWriteHive(self.result, self.outputUrl1)


class PyEvaluateClustering:
    pass
