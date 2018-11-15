from ..data import data


class PyCrossValidate:
    pass


class PyEvaluateBinaryClassifier:
    def __init__(self, args):
        self.inputUrl1 = args["input1"][0]["value"]
        self.inputUrl2 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]

        self.originalDF = None
        self.transformDF = None
        self.result = None
    def getIn(self):
        if self.inputUrl1.endswith(".pkl"):
            print("using scikit-learn")
            import pandas as pd
            from sklearn.externals import joblib
            # self.originalDF = data.PyReadHive(self.inputUrl2)
            self.originalDF = data.PyReadCSV(self.inputUrl2)

            self.model = joblib.load(self.inputUrl1)
        else:
            print("using PySpark")
            from pyspark.sql import SparkSession
            from pyspark.ml import PipelineModel

            self.spark = SparkSession \
                .builder \
                .config("spark.sql.warehouse.dir", "hdfs://10.110.18.216/user/hive/warehouse") \
                .enableHiveSupport() \
                .getOrCreate()
            self.spark.sql("use sparktest")
            self.originalDF = data.SparkReadHive(self.inputUrl2, self.spark)
            self.model = PipelineModel.load(self.inputUrl1)

    def execute(self):
        featureCols = self.param["features"].split(",")
        labelCol = self.param["label"].split(",")[0]
        if self.inputUrl1.endswith(".pkl"):
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
                {"accuracy": [accuracy], "precision": [precision], "recall": [recall], "f1 score": [f1]})

        else:
            print("using PySpark")
            from pyspark.sql.types import StructType, StructField, FloatType

            self.transformDF = self.model.transform(self.originalDF).drop("features")
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
            self.result = self.spark.createDataFrame(self.spark.sparkContext.parallelize([(accuracy, precision, recall, f1)]),
                                                     result_struct)
            # output
    def setOut(self):
        if self.inputUrl1.endswith(".pkl"):
            data.PyWriteCSV(self.result, self.outputUrl1)
            # data.PyWriteHive(self.result, self.outputUrl1)

        else:
            data.SparkWriteHive(self.result, self.outputUrl1)


class PyEvaluateMultiLabelClassifier:
    pass


class PyEvaluateRegressor:
    pass


class PyEvaluateClustering:
    pass
