class PyJoin:
    pass
class PyMergeRow:
    pass
class PyMergeColumn:
    pass
class PySelectColumn:
    pass
class PyGroupBy:
    pass
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

class PyReplace:
    pass
class PyAsColumn:
    pass
class PyTranspose:
    pass
class PyTable2KV:
    pass
class PyKV2Table:
    pass
class PyHandleMissingValue:
    pass
class PyChangeColumnName:
    pass
class PyAddId:
    pass
class PyDropDuplicate:
    pass
class PySortValue:
    pass
class RFMProcess:
    pass
class RFMAnalysis:
    pass
class SparkJoin:
    pass
class SparkMergeRow:
    pass
class SparkMergeColumn:
    pass
class SparkSelectColumn:
    pass
class SparkGroupBy:
    pass
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

class SparkReplace:
    pass
class SparkAsColumn:
    pass
class SparkTranspose:
    pass
class SparkTable2KV:
    pass
class SparkKV2Table:
    pass
class SparkHandleMissingValue:
    pass
class SparkChangeColumnName:
    pass
class SparkAddId:
    pass
class SparkDropDuplicate:
    pass
class SparkSortValue:
    pass
