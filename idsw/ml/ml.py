class PyTrainModel:
    def __init__(self, args):
        self.inputUrl1 = args["Dinput1"]
        self.inputUrl2 = args["Dinput2"]
        self.outputUrl1 = args["Doutput1"]
        self.featureCols = args["Dfeatures"].split(",")
        try:
            self.labelCol = args["Dlabel"].split(",")[0]
        except KeyError:
            self.labelCol=None

    def getIn(self):
        if self.inputUrl1.endswith(".pkl"):
            print("using scikit-learn")
            import pandas as pd
            from sklearn.externals import joblib
            self.originalDF = pd.read_csv(self.inputUrl2, encoding="utf-8")
            self.model = joblib.load(self.inputUrl1)
        else:
            print("using PySpark")
            from pyspark.sql import SparkSession
            from pyspark.ml import Pipeline, PipelineModel

            self.spark = SparkSession \
                .builder \
                .config("spark.sql.warehouse.dir", "hdfs://10.110.18.216/user/hive/warehouse") \
                .enableHiveSupport() \
                .getOrCreate()
            self.spark.sql("use sparktest")
            self.originalDF = self.spark.sql("select * from " + self.inputUrl2)
            self.model = Pipeline.load(self.inputUrl1).getStages()[0]

    def execute(self):
        if self.inputUrl1.endswith(".pkl"):
            import pandas as pd
            import sklearn.cluster
            if (not isinstance(self.model, sklearn.cluster.k_means_.KMeans)) & (
            not isinstance(self.model, sklearn.cluster.dbscan_.DBSCAN)):
                self.model.fit(self.originalDF[self.featureCols].as_matrix(), self.originalDF[self.labelCol].values)
            else:
                self.model.fit(self.originalDF[self.featureCols])
        else:
            print("using PySpark")
            from pyspark.sql import SparkSession
            from pyspark.ml import Pipeline
            import pyspark.ml.clustering
            from pyspark.ml.feature import VectorAssembler
            if (not isinstance(self.model, pyspark.ml.clustering.KMeans)):
                self.vectorAssembler = VectorAssembler(inputCols=self.featureCols, outputCol="features")
                self.model.setParams(featuresCol="features", labelCol=self.labelCol)
                self.pipeline = Pipeline(stages=[self.vectorAssembler, self.model])
                self.pipelinemodel = self.pipeline.fit(self.originalDF)

    def setOut(self):
        if self.inputUrl1.endswith(".pkl"):
            from sklearn.externals import joblib
            joblib.dump(self.model, self.outputUrl1, compress=True)

        else:
            self.pipelinemodel.write().overwrite().save(self.outputUrl1)

class PyRandomForest:
    def __init__(self, args):
        self.outputUrl1 = args["Doutput1"]
        self.n_estimators = int(args["DtreeNum"])
        self.criterion = args["Dcriterion"]
        self.max_depth = int(args["DmaxDepth"])
        self.min_samples_split = int(args["DminSamplesSplit"])
        self.min_samples_leaf = int(args["DminSamplesLeaf"])

    def getIn(self):
        pass

    def execute(self):
        print("using scikit-learn")
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth,
                                            min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)

    def setOut(self):
        from sklearn.externals import joblib
        joblib.dump(self.model, self.outputUrl1, compress=True)

class SparkRandomForest:
    def __init__(self, args):
        self.outputUrl1 = args["Doutput1"]
        self.n_estimators = int(args["DtreeNum"])
        self.criterion = args["Dcriterion"]
        self.max_depth = int(args["DmaxDepth"])
        self.min_samples_split = int(args["DminSamplesSplit"])
        self.min_samples_leaf = int(args["DminSamplesLeaf"])

        from pyspark.sql import SparkSession

        self.spark = SparkSession \
            .builder \
            .config("spark.sql.warehouse.dir", "hdfs://10.110.18.216/user/hive/warehouse") \
            .enableHiveSupport() \
            .getOrCreate()
        self.spark.sql("use sparktest")

    def getIn(self):
        pass

    def execute(self):
        from pyspark.ml.classification import RandomForestClassifier
        from pyspark.ml import Pipeline

        self.model = Pipeline(stages=[RandomForestClassifier(numTrees=self.n_estimators, impurity=self.criterion, maxDepth=self.max_depth, minInstancesPerNode=self.min_samples_leaf)])

    def setOut(self):

        self.model.write().overwrite().save(self.outputUrl1)

class PyEvaluateBinaryClassifier:
    def __init__(self, args):
        self.inputUrl1 = args["Dinput1"]
        self.inputUrl2 = args["Dinput2"]
        self.outputUrl1 = args["Doutput1"]
        self.featureCols = args["Dfeatures"].split(",")
        try:
            self.labelCol = args["Dlabel"].split(",")[0]
        except AttributeError:
            self.labelCol=None

    def getIn(self):
        if self.inputUrl1.endswith(".pkl"):
            print("using scikit-learn")
            import pandas as pd
            from sklearn.externals import joblib
            self.originalDF = pd.read_csv(self.inputUrl2, encoding="utf-8")
            self.model = joblib.load(self.inputUrl1)
        else:
            print("using PySpark")
            from pyspark.sql import SparkSession
            from pyspark.ml import Pipeline, PipelineModel

            self.spark = SparkSession \
                .builder \
                .config("spark.sql.warehouse.dir", "hdfs://10.110.18.216/user/hive/warehouse") \
                .enableHiveSupport() \
                .getOrCreate()
            self.spark.sql("use sparktest")
            self.originalDF = self.spark.sql("select * from " + self.inputUrl2)
            self.model = PipelineModel.load(self.inputUrl1)

    def execute(self):
        if self.inputUrl1.endswith(".pkl"):
            import pandas as pd
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            self.transformDF = self.originalDF.copy()
            self.predicted = self.model.predict(self.originalDF[self.featureCols])

            self.transformDF["prediction"] = self.predicted

            self.accuracy = accuracy_score(self.originalDF[self.labelCol], self.predicted)
            self.precision = precision_score(self.originalDF[self.labelCol], self.predicted)
            self.recall = recall_score(self.originalDF[self.labelCol], self.predicted)
            self.f1 = f1_score(self.originalDF[self.labelCol], self.predicted)

            self.result = pd.DataFrame.from_dict(
                {"accuracy": [self.accuracy], "precision": [self.precision], "recall": [self.recall], "f1 score": [self.f1]})

        else:
            print("using PySpark")
            from pyspark.sql import SparkSession
            from pyspark.ml import Pipeline
            from pyspark.sql.types import StructType, StructField, FloatType

            self.transformDF = self.model.transform(self.originalDF).drop("features")
            TP = float(self.transformDF.filter(self.transformDF.prediction == 1.0).filter(self.transformDF[self.labelCol] == self.transformDF.prediction).count())
            TN = float(self.transformDF.filter(self.transformDF.prediction == 0.0).filter(self.transformDF[self.labelCol] == self.transformDF.prediction).count())
            FN = float(self.transformDF.filter(self.transformDF.prediction == 0.0).filter(~(self.transformDF[self.labelCol] == self.transformDF.prediction)).count())
            FP = float(self.transformDF.filter(self.transformDF.prediction == 1.0).filter(~(self.transformDF[self.labelCol] == self.transformDF.prediction)).count())

            self.accuracy = (TP + TN) / (TP + FP + TN + FN)
            self.precision = TP / (TP + FP)
            self.recall = TP / (TP + FN)
            self.f1 = (2 * self.precision * self.recall) / (self.precision + self.recall)
            print("accuracy: %s" % self.accuracy)
            print("precision: %s" % self.precision)
            print("recall: %s" % self.recall)
            print("f1 score: %s" % self.f1)

            self.result_struct = StructType(
                [StructField("accuracy", FloatType(), True), StructField("precision", FloatType(), True),
                 StructField("recall", FloatType(), True), StructField("f1 score", FloatType(), True)])
            self.result = self.spark.createDataFrame(self.spark.sparkContext.parallelize([(self.accuracy, self.precision, self.recall, self.f1)]),
                                                     self.result_struct)
            # output
    def setOut(self):
        if self.inputUrl1.endswith(".pkl"):
            self.result.to_csv(self.outputUrl1, index=False)

        else:
            self.result.write.mode("overwrite").format("hive").saveAsTable(self.outputUrl1)
