from ..data import data


class PyTrainModel:
    def __init__(self, args):
        self.originalDF = None
        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]

    def getIn(self):
        if self.inputUrl1.endswith(".pkl"):
            print("using scikit-learn")

            from sklearn.externals import joblib
            self.originalDF = data.PyReadCSV(self.inputUrl2)
            # self.originalDF = data.PyReadHive(self.inputUrl2)
            self.model = joblib.load(self.inputUrl1)
        else:
            print("using PySpark")
            from pyspark.sql import SparkSession
            from pyspark.ml import Pipeline

            self.spark = SparkSession \
                .builder \
                .config("spark.sql.warehouse.dir", "hdfs://10.110.18.216/user/hive/warehouse") \
                .enableHiveSupport() \
                .getOrCreate()
            self.spark.sql("use sparktest")
            self.originalDF = data.SparkReadHive(self.inputUrl2, self.spark)
            self.model = Pipeline.load(self.inputUrl1).getStages()[0]

    def execute(self):
        featureCols = self.param["features"].split(",")
        labelCol = self.param["label"].split(",")[0]

        if self.inputUrl1.endswith(".pkl"):

            import sklearn.cluster
            if (not isinstance(self.model, sklearn.cluster.k_means_.KMeans)) & (
                    not isinstance(self.model, sklearn.cluster.dbscan_.DBSCAN)):
                self.model.fit(self.originalDF[featureCols], self.originalDF[labelCol])
            else:
                self.model.fit(self.originalDF[featureCols])
        else:
            print("using PySpark")
            from pyspark.sql import SparkSession
            from pyspark.ml import Pipeline
            import pyspark.ml.clustering
            from pyspark.ml.feature import VectorAssembler
            if not isinstance(self.model, pyspark.ml.clustering.KMeans):
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
