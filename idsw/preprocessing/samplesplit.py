from ..data import data


class PySampleData:
    pass


class PySplitData:
    def __init__(self, args):
        """
        Standalone version for split data, including byRatio and byThreshold
        @param args: dict
        splitBy: string one of byRation and byThreshold
        ratio: double
        thresholdColumn: stirng
        threshold: double
        """
        self.originalDF = None
        self.DF1 = None
        self.DF2 = None
        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = args["output"][1]["value"]
        self.param = args["param"]

    def getIn(self):

        self.originalDF = data.PyReadCSV(self.inputUrl1)
        # self.originalDF = data.PyReadHive(self.inputUrl1)

    def execute(self):

        def splitByRatio():
            from sklearn.model_selection import train_test_split
            ratio = float(self.param["ratio"])
            self.DF1, self.DF2 = train_test_split(self.originalDF, train_size=ratio)

        def splitByThreshold():
            thresholdColumn = self.param["thresholdColumn"]
            threshold = float(self.param["threshold"])
            self.DF1 = self.originalDF[self.originalDF[thresholdColumn] >= threshold]
            self.DF2 = self.originalDF[self.originalDF[thresholdColumn] < threshold]

        mode = self.param["splitBy"]
        modes = {"byRatio": splitByRatio, "byThreshold": splitByThreshold}
        modes[mode]()

    def setOut(self):

        # data.PyWriteHive(self.DF1, self.outputUrl1)
        # data.PyWriteHive(self.DF2, self.outputUrl2)
        data.PyWriteCSV(self.DF1, self.outputUrl1)
        data.PyWriteCSV(self.DF2, self.outputUrl2)


class SparkSampleData:
    pass


class SparkSplitData:

    def __init__(self, args):
        """
         Standalone version for split data, including byRatio and byThreshold
         @param args: dict
         splitBy: string one of byRation and byThreshold
         ratio: double
         thresholdColumn: stirng
         threshold: double
         """
        self.originalDF = None
        self.DF1 = None
        self.DF2 = None

        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = args["output"][1]["value"]

        self.param = args["param"]

        print("using PySpark")
        from pyspark.sql import SparkSession

        self.spark = SparkSession \
            .builder \
            .config("spark.sql.warehouse.dir", "hdfs://10.110.18.216/user/hive/warehouse") \
            .enableHiveSupport() \
            .getOrCreate()

    def getIn(self):

        self.originalDF = data.SparkReadHive(self.inputUrl1, self.spark)

    def execute(self):

        def splitByRatio():
            ratio = float(self.param["ratio"])
            self.DF1, self.DF2 = self.originalDF.randomSplit([ratio, 1-ratio])

        def splitByThreshold():
            thresholdColumn = self.param["thresholdColumn"]
            threshold = float(self.param["threshold"])
            self.DF1 = self.originalDF.filter(self.originalDF[thresholdColumn] >= threshold)
            self.DF2 = self.originalDF.filter(self.originalDF[thresholdColumn] < threshold)

        mode = self.param["splitBy"]
        modes = {"byRatio": splitByRatio, "byThreshold": splitByThreshold}
        modes[mode]()

    def setOut(self):
        data.PyWriteHive(self.DF1, self.outputUrl1)
        data.PyWriteHive(self.DF2, self.outputUrl2)
