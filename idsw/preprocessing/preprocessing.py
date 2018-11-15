class PyTypeTransform:
    def __init__(self, args):

        self.inputUrl1 = args["Dinput1"]
        self.outputUrl1 = args["Doutput1"]
        try:
            self.toDoubleColumns = args["DtoDouble"].split(",")
        except KeyError:
            self.toDoubleColumns = None
        try:
            self.defaultDoubleValue = args["DdefaultDoubleValue"]
        except KeyError:
            self.defaultDoubleValue = 0.0
        try:
            self.toIntColumns = args["DtoInt"].split(",")
        except KeyError:
            self.toIntColumns = None
        try:
            self.defaultIntValue = args["DdefaultIntValue"]
        except KeyError:
            self.defaultDoubleValue = 0.0
        try:
            self.toCategoricalColumns = args["DtoCategoricalColumns"].split(",")
        except KeyError:
            self.toCategoricalColumns = None

    def getIn(self):
        import pandas as pd
        import sys
        try:
            self.originalDF = pd.read_csv(self.inputUrl1)
        except:
            print("plz set inputUrl1")
            sys.exit(0)

    def execute(self):
        import sys
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        self.transformedDF = self.originalDF.copy()
        if self.toCategoricalColumns != None:
            self.transformedDF[self.toCategoricalColumns] = self.transformedDF[self.toCategoricalColumns].apply(LabelEncoder().fit_transform)

    def setOut(self):
        self.transformedDF.to_csv(self.outputUrl1, index=False, encoding='utf-8')

class SparkTypeTransform:
    def __init__(self, args):
        self.inputUrl1 = args["Dinput1"]
        self.outputUrl1 = args["Doutput1"]
        try:
            self.toDoubleColumns = args["DtoDouble"].split(",")
        except KeyError:
            self.toDoubleColumns = None
        try:
            self.defaultDoubleValue = args["DdefaultDoubleValue"]
        except KeyError:
            self.defaultDoubleValue = 0.0
        try:
            self.toIntColumns = args["DtoInt"].split(",")
        except KeyError:
            self.toIntColumns = None
        try:
            self.defaultIntValue = args["DdefaultIntValue"]
        except KeyError:
            self.defaultDoubleValue = 0.0
        try:
            self.toCategoricalColumns = args["DtoCategoricalColumns"].split(",")
        except KeyError:
            self.toCategoricalColumns = None

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

        from pyspark.ml.feature import StringIndexer
        indexers = [StringIndexer(inputCol=column, outputCol="index_" + column).fit(self.originalDF) for column in self.toCategoricalColumns]
        self.transformedDF = self.originalDF
        for i in range(len(indexers)):
            self.transformedDF = indexers[i].transform(self.transformedDF).drop(self.toCategoricalColumns[i]).withColumnRenamed("index_" + self.toCategoricalColumns[i], self.toCategoricalColumns[i])

    def setOut(self):
        self.transformedDF.write.mode("overwrite").format("hive").saveAsTable(self.outputUrl1)

class PyMinMaxScaler:
    def __init__(self, args):
        from sklearn.preprocessing import MinMaxScaler
        self.inputUrl1 = args["Dinput1"]
        self.inputUrl2 = args["Dinput2"]
        self.outputUrl1 = args["Doutput1"]
        self.outputUrl2 = args["Doutput2"]
        try:
            self.columns = args["Dcolumns"].split(",")
        except KeyError as e:
            self.columns = None
        if self.columns == "":
            self.columns = None
        self.scaler = MinMaxScaler()

    def getIn(self):
        import pandas as pd
        import sys
        try:
            self.originalDF = pd.read_csv(self.inputUrl1)
        except:
            print("plz set inputUrl1")
            sys.exit(0)

        try:
            self.parameterDF = pd.read_csv(self.inputUrl2)
        except:
            print("will execute fit_transform")
    def execute(self):
        import sys
        import pandas as pd
        print(self.inputUrl2)
        print(self.columns)
        if (self.inputUrl2 == "-Doutput1") & (self.columns!= None):
            print("fit_transform")
            self.transformedDF = self.originalDF.copy()
            self.transformedDF[self.columns] = self.scaler.fit_transform(self.originalDF[self.columns])
            self.parameterDF = pd.DataFrame([self.scaler.data_min_, self.scaler.data_max_], columns=self.columns)
        elif self.inputUrl2 != "-Doutput1'":
            self.transformedDF = self.originalDF.copy()
            for self.col in self.parameterDF.columns:
                self.p_min = self.parameterDF.loc[0, self.col]
                self.p_max = self.parameterDF.loc[1, self.col]
                self.transformedDF[self.col] = (self.transformedDF[self.col] - self.p_min) / (self.p_max - self.p_min)
        else:
            sys.exit(0)
    def setOut(self):
        self.transformedDF.to_csv(self.outputUrl1, index=False, encoding='utf-8')
        self.parameterDF.to_csv(self.outputUrl2, index=False, encoding='utf-8')

class SparkMinMaxScaler:
    def __init__(self, args):
        self.inputUrl1 = args["Dinput1"]
        self.inputUrl2 = args["Dinput2"]
        self.outputUrl1 = args["Doutput1"]
        self.outputUrl2 = args["Doutput2"]
        try:
            self.columns = args["Dcolumns"].split(",")
        except AttributeError as e:
            self.columns = None
        if self.columns == "":
            self.columns = None

        print("using PySpark")
        from pyspark.sql import SparkSession

        self.spark = SparkSession \
            .builder \
            .config("spark.sql.warehouse.dir", "hdfs://10.110.18.216/user/hive/warehouse") \
            .enableHiveSupport() \
            .getOrCreate()

    def getIn(self):

        self.originalDF = self.spark.sql("select * from " + self.inputUrl1)
        if self.inputUrl2 == "":
            print("execute fit_transform")
        else:
            self.parameterDF = self.spark.sql("select * from " + self.inputUrl2)
            print("execute transform")

    def execute(self):
        from pyspark.sql import functions
        if self.inputUrl2 == "":
            self.mmParamList = []
            self.transformedDF = self.originalDF
            for self.col in self.columns:
                self.mmRow = self.originalDF.select(functions.min(self.col), functions.max(self.col)).first()
                self.mmMin = self.mmRow["min(" + self.col + ")"]
                self.mmMax = self.mmRow["max(" + self.col + ")"]
                self.mmParamList.append([self.col, float(self.mmMin), float(self.mmMax)])
                self.transformedDF = self.transformedDF.withColumn(self.col + "mm", (functions.col(self.col) - self.mmMin) / (self.mmMax - self.mmMin))\
                    .drop(self.col).withColumnRenamed(self.col + "mm", self.col)

            self.parameterDF = self.spark.createDataFrame(self.mmParamList, ['col','min','max'])

        else:
            self.transformedDF = self.originalDF
            self.parameterDFP = self.parameterDF.toPandas()
            for self.col in self.parameterDFP["col"]:
                self.mmParamList = [self.parameterDFP.loc[self.parameterDFP["col"] == self.col, "min"].tolist()[0], self.parameterDFP.loc[self.parameterDFP["col"] == self.col, "max"].tolist()[0]]
                self.transformedDF = self.transformedDF.withColumn(self.col+"mm", (functions.col(self.col) - self.mmParamList[0])/(self.mmParamList[1]-self.mmParamList[0])).drop(self.col).withColumnRenamed(self.col+"mm", self.col)

    def setOut(self):
        self.transformedDF.write.mode("overwrite").format("hive").saveAsTable(self.outputUrl1)
        self.parameterDF.write.mode("overwrite").format("hive").saveAsTable(self.outputUrl2)

class PyStandardScaler:
    def __init__(self, args):
        from sklearn.preprocessing import StandardScaler
        self.inputUrl1 = args["Dinput1"]
        self.inputUrl2 = args["Dinput2"]
        self.outputUrl1 = args["Doutput1"]
        self.outputUrl2 = args["Doutput2"]
        try:
            self.columns = args["Dcolumns"].split(",")
        except AttributeError as e:
            self.columns = None
        if self.columns == "":
            self.columns = None
        self.standardScaler = StandardScaler()

    def getIn(self):
        import pandas as pd
        import sys
        try:
            self.originalDF = pd.read_csv(self.inputUrl1)
        except:
            print("plz set inputUrl1")
            sys.exit(0)
        if self.inputUrl2 == "":
            print("will execute fit_transform")
        else:
            self.parameterDF = pd.read_csv(self.inputUrl2)

    def execute(self):
        import sys
        import pandas as pd

        if (self.inputUrl2 == "") & (self.columns!= None):
            self.transformedDF = self.originalDF.copy()
            self.transformedDF[self.columns] = self.standardScaler.fit_transform(self.originalDF[self.columns])
            self.parameterDF = pd.DataFrame([self.standardScaler.mean_, self.standardScaler.var_], columns=self.columns)
        elif self.inputUrl2 != "":
            self.transformedDF = self.originalDF.copy()
            for self.col in self.parameterDF.columns:
                self.p_mean = self.parameterDF.loc[0, self.col]
                self.p_std = self.parameterDF.loc[1, self.col]
                self.transformedDF[self.col] = (self.transformedDF[self.col] - self.p_mean) / (self.p_std)
        else:
            sys.exit(0)
    def setOut(self):
        self.transformedDF.to_csv(self.outputUrl1, index=False, encoding='utf-8')
        self.parameterDF.to_csv(self.outputUrl2, index=False, encoding='utf-8')

class SparkStandardScaler:
    def __init__(self, args):
        self.inputUrl1 = args["Dinput1"]
        self.inputUrl2 = args["Dinput2"]
        self.outputUrl1 = args["Doutput1"]
        self.outputUrl2 = args["Doutput2"]
        try:
            self.columns = args["Dcolumns"].split(",")
        except AttributeError as e:
            self.columns = None
        if self.columns == "":
            self.columns = None

        print("using PySpark")
        from pyspark.sql import SparkSession

        self.spark = SparkSession \
            .builder \
            .config("spark.sql.warehouse.dir", "hdfs://10.110.18.216/user/hive/warehouse") \
            .enableHiveSupport() \
            .getOrCreate()

    def getIn(self):

        self.originalDF = self.spark.sql("select * from " + self.inputUrl1)
        if self.inputUrl2 == "":
            print("execute fit_transform")
        else:
            self.parameterDF = self.spark.sql("select * from " + self.inputUrl2)
            print("execute transform")

    def execute(self):
        from pyspark.sql import functions
        if self.inputUrl2 == "":
            self.mmParamList = []
            self.transformedDF = self.originalDF
            for self.col in self.columns:
                self.mmRow = self.originalDF.select(functions.avg(self.col), functions.stddev(self.col)).first()
                self.mmAvg = self.mmRow["avg(" + self.col + ")"]
                self.mmStd = self.mmRow["stddev_samp(" + self.col + ")"]
                self.mmParamList.append([self.col, float(self.mmAvg), float(self.mmStd)])
                self.transformedDF = self.transformedDF.withColumn(self.col + "ss", (functions.col(self.col) - self.mmAvg) / (self.mmStd))\
                    .drop(self.col).withColumnRenamed(self.col + "ss", self.col)

            self.parameterDF = self.spark.createDataFrame(self.mmParamList, ['col','avg','std'])

        else:
            self.transformedDF = self.originalDF
            self.parameterDFP = self.parameterDF.toPandas()
            for self.col in self.parameterDFP["col"]:
                self.mmParamList = [self.parameterDFP.loc[self.parameterDFP["col"] == self.col, "avg"].tolist()[0], self.parameterDFP.loc[self.parameterDFP["col"] == self.col, "std"].tolist()[0]]
                self.transformedDF = self.transformedDF.withColumn(self.col+"ss", (functions.col(self.col) - self.mmParamList[0])/(self.mmParamList[1]-self.mmParamList[0]))\
                    .drop(self.col).withColumnRenamed(self.col+"ss", self.col)

    def setOut(self):
        self.transformedDF.write.mode("overwrite").format("hive").saveAsTable(self.outputUrl1)
        self.parameterDF.write.mode("overwrite").format("hive").saveAsTable(self.outputUrl2)

class PySplitData:
    def __init__(self, args):
        self.inputUrl1 = args["Dinput1"]
        self.outputUrl1 = args["Doutput1"]
        self.outputUrl2 = args["Doutput2"]
        self.ratio = float(args["Dratio"])

    def getIn(self):
        import pandas as pd
        import sys
        try:
            self.originalDF = pd.read_csv(self.inputUrl1)
        except:
            print("plz set inputUrl1")
            sys.exit(0)

    def execute(self):

        from sklearn.model_selection import train_test_split

        self.trainDF, self.testDF = train_test_split(self.originalDF, train_size=self.ratio)

    def setOut(self):

        self.trainDF.to_csv(self.outputUrl1, index=False, encoding='utf-8')
        self.testDF.to_csv(self.outputUrl2, index=False, encoding='utf-8')

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
