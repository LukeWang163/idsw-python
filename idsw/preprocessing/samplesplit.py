class PySampleData:
    pass

class PySplitData:
    def __init__(self, args):
        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = args["output"][1]["value"]
        self.param = args["param"]

    def getIn(self):
        from ..data import data
        self.originalDF = data.PyReadCSV(self.inputUrl1)

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

        self.DF1.to_csv(self.outputUrl1, index=False, encoding='utf-8')
        self.DF2.to_csv(self.outputUrl2, index=False, encoding='utf-8')

class SparkSampleData:
    pass

class SparkSplitData:
    def __init__(self, args):
        self.inputUrl1 = args["Dinput1"]
        self.outputUrl1 = args["Doutput1"]
        self.outputUrl2 = args["Doutput2"]
        self.ratio = float(args["Dratio"])

        print("using PySpark")
        from pyspark.sql import SparkSession

        self.spark = SparkSession \
            .builder \
            .config("spark.sql.warehouse.dir", "hdfs://10.110.18.216/user/hive/warehouse") \
            .enableHiveSupport() \
            .getOrCreate()

    def getIn(self):

        self.originalDF = self.spark.sql("select * from " + self.inputUrl1)

    def execute(self):

        self.trainDF, self.testDF = self.originalDF.randomSplit([self.ratio, 1-self.ratio])


    def setOut(self):
        self.trainDF.write.mode("overwrite").format("hive").saveAsTable(self.outputUrl1)
        self.testDF.write.mode("overwrite").format("hive").saveAsTable(self.outputUrl2)
