from ..data import data


class PyGroupIntoBins:
    pass


class PyNormalizeData:
    def __init__(self, args):
        self.transformedDF = None
        self.parameterDF = None
        self.inputUrl1 = args["input"][0]["value"]
        try:
            self.inputUrl2 = args["input"][1]["value"]
        except IndexError:
            self.inputUrl2 = ""
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = args["output"][1]["value"]
        try:
            self.columns = args["columns"].split(",")
        except KeyError as e:
            self.columns = None
        self.param = args["param"]

    def getIn(self):
        self.originalDF = data.PyReadCSV(self.inputUrl1)

        if self.outputUrl2 == "":
            print("will execute fit_transform")
        else:
            self.parameterDF = data.PyReadCSV(self.inputUrl2)

    def execute(self):
        import pandas as pd

        def minMax():
            if (self.parameterDF is None) & (self.columns is not None):
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                self.transformedDF = self.originalDF.copy()
                self.transformedDF[self.columns] = scaler.fit_transform(self.originalDF[self.columns])
                self.parameterDF = pd.DataFrame([scaler.data_min_, scaler.data_max_], columns=self.columns)
            elif self.parameterDF is not None:
                self.transformedDF = self.originalDF.copy()
                for col in self.parameterDF.columns:
                    p_min = self.parameterDF.loc[0, col]
                    p_max = self.parameterDF.loc[1, col]
                    self.transformedDF[col] = (self.transformedDF[col] - p_min) / (p_max - p_min)

        def standard():
            if (self.parameterDF is None) & (self.columns is not None):
                from sklearn.preprocessing import StandardScaler
                standardScaler = StandardScaler()
                self.transformedDF = self.originalDF.copy()
                self.transformedDF[self.columns] = standardScaler.fit_transform(self.originalDF[self.columns])
                self.parameterDF = pd.DataFrame([standardScaler.mean_, standardScaler.var_],
                                                columns=self.columns)

            elif self.parameterDF is not None:
                self.transformedDF = self.originalDF.copy()
                for col in self.parameterDF.columns:
                    p_mean = self.parameterDF.loc[0, col]
                    p_std = self.parameterDF.loc[1, col]
                    self.transformedDF[col] = (self.transformedDF[col] - p_mean) / p_std

        mode = self.param["mode"]
        modes = {"minMax": minMax, "standard": standard}
        modes[mode]()

    def setOut(self):
        data.PyWriteCSV(self.transformedDF, self.outputUrl1)
        data.PyWriteCSV(self.parameterDF, self.outputUrl2)


class SparkGroupIntoBins:
    pass


class SparkNormalizeData:
    def __init__(self, args):
        self.transformedDF = None
        self.parameterDF = None
        self.inputUrl1 = args["input"][0]["value"]
        try:
            self.inputUrl2 = args["input"][1]["value"]
        except IndexError:
            self.inputUrl2 = ""
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = args["output"][1]["value"]
        try:
            self.columns = args["columns"].split(",")
        except KeyError as e:
            self.columns = None
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
        from pyspark.sql import functions

        def minMax():
            if (self.parameterDF is None) & (self.columns is not None):
                mmParamList = []
                self.transformedDF = self.originalDF
                for col in self.columns:
                    mmRow = self.originalDF.select(functions.min(col), functions.max(col)).first()
                    mmMin = mmRow["min(" + col + ")"]
                    mmMax = mmRow["max(" + col + ")"]
                    mmParamList.append([col, float(mmMin), float(mmMax)])
                    self.transformedDF = self.transformedDF\
                        .withColumn(col + "mm", (functions.col(col) - mmMin) / (mmMax - mmMin)) \
                        .drop(col)\
                        .withColumnRenamed(col + "mm", col)

                self.parameterDF = self.spark.createDataFrame(mmParamList, ['col', 'min', 'max'])

            elif self.parameterDF is not None:
                self.transformedDF = self.originalDF
                self.parameterDFP = self.parameterDF.toPandas()
                for col in self.parameterDFP["col"]:
                    mmParamList = [self.parameterDFP.loc[self.parameterDFP["col"] == col, "min"].tolist()[0],
                                        self.parameterDFP.loc[self.parameterDFP["col"] == col, "max"].tolist()[0]]
                    self.transformedDF = self.transformedDF\
                        .withColumn(col + "mm", (functions.col(col) - mmParamList[0]) / (mmParamList[1] - mmParamList[0]))\
                        .drop(col)\
                        .withColumnRenamed(col + "mm", col)

        def standard():
            if (self.parameterDF is None) & (self.columns is not None):
                mmParamList = []
                self.transformedDF = self.originalDF
                for col in self.columns:
                    mmRow = self.originalDF.select(functions.avg(col), functions.stddev(col)).first()
                    mmAvg = mmRow["avg(" + col + ")"]
                    mmStd = mmRow["stddev_samp(" + col + ")"]
                    mmParamList.append([col, float(mmAvg), float(mmStd)])
                    self.transformedDF = self.transformedDF\
                        .withColumn(col + "ss",(functions.col(col) - mmAvg) / mmStd) \
                        .drop(col)\
                        .withColumnRenamed(col + "ss", col)

                self.parameterDF = self.spark.createDataFrame(mmParamList, ['col', 'avg', 'std'])

            elif self.parameterDF is not None:
                self.transformedDF = self.originalDF
                self.parameterDFP = self.parameterDF.toPandas()
                for col in self.parameterDFP["col"]:
                    mmParamList = [self.parameterDFP.loc[self.parameterDFP["col"] == col, "avg"].tolist()[0],
                                   self.parameterDFP.loc[self.parameterDFP["col"] == col, "std"].tolist()[0]]
                    self.transformedDF = self.transformedDF\
                        .withColumn(col+"ss", (functions.col(col) - mmParamList[0])/(mmParamList[1] - mmParamList[0])) \
                        .drop(col)\
                        .withColumnRenamed(col + "ss", col)

        mode = self.param["mode"]
        modes = {"minMax": minMax, "standard": standard}
        modes[mode]()

    def setOut(self):

        data.SparkWriteHive(self.transformedDF, self.outputUrl1)
        data.SparkWriteHive(self.parameterDF, self.outputUrl2)

