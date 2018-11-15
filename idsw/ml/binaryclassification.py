class PySVM:
    pass


class PyDecisionTree:
    pass


class PyRandomForest:
    def __init__(self, args):
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]
        self.model = None

    def getIn(self):
        return

    def execute(self):
        print("using scikit-learn")
        n_estimators = int(self.param["treeNum"])
        criterion = self.param["criterion"]
        max_depth = int(self.param["maxDepth"])
        min_samples_split = int(self.param["minSamplesSplit"])
        min_samples_leaf = int(self.param["minSamplesLeaf"])
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    def setOut(self):
        from sklearn.externals import joblib
        joblib.dump(self.model, self.outputUrl1, compress=True)


class PyGBDT:
    pass


class PyLogisticRegression:
    pass


class PyNaiveBayes:
    pass


class PyBayesNetwork:
    pass


class PyAveragedPerceptron:
    pass


class PyMLP:
    pass


class PyAutoML:
    pass


class SparkRandomForest:
    def __init__(self, args):
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]
        self.model = None

        print("using PySpark")
        from pyspark.sql import SparkSession

        self.spark = SparkSession \
            .builder \
            .config("spark.sql.warehouse.dir", "hdfs://10.110.18.216/user/hive/warehouse") \
            .enableHiveSupport() \
            .getOrCreate()

    def getIn(self):
        return

    def execute(self):
        from pyspark.ml.classification import RandomForestClassifier
        from pyspark.ml import Pipeline

        n_estimators = int(self.param["treeNum"])
        criterion = self.param["criterion"]
        max_depth = int(self.param["maxDepth"])
        min_samples_split = int(self.param["minSamplesSplit"])
        min_samples_leaf = int(self.param["minSamplesLeaf"])

        self.model = Pipeline(stages=[
            RandomForestClassifier(numTrees=n_estimators, impurity=criterion,
                                   maxDepth=max_depth,
                                   minInstancesPerNode=min_samples_leaf)])

    def setOut(self):
        self.model.write().overwrite().save(self.outputUrl1)

