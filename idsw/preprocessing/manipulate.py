#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.preprocessing.manipulate.py
# @Desc    : Scripts for manipulating data. 数据预处理->数据操作
from .. import utils


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

    def __init__(self, args, args2):
        """
        Standalone version for type transformation
        @param args: dict
        toDoubleColumns: list
        defaultDoubleValue: double
        toIntColumns: list
        defaultIntValue: int
        toCategoricalColumns: list
        """
        self.originalDF = None
        self.transformedDF = None
        self.paramDF = None

        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = self.outputUrl1+"toCategorical"
        self.param = args["param"]
        try:
            self.toDoubleColumns = self.param["toDouble"]
        except KeyError:
            self.toDoubleColumns = None
        try:
            self.defaultDoubleValue = self.param["defaultDoubleValue"]
        except KeyError:
            self.defaultDoubleValue = 0.0
        try:
            self.toIntColumns = self.param["toInt"]
        except KeyError:
            self.toIntColumns = None
        try:
            self.defaultIntValue = self.param["defaultIntValue"]
        except KeyError:
            self.defaultIntValue = 0
        try:
            self.toCategoricalColumns = self.param["toCategoricalColumns"]
        except KeyError:
            self.toCategoricalColumns = None

        self.mode = (self.param["mode"])

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):

        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl1)
        # self.originalDF = self.dataUtil.PyReadCSV(self.inputUrl1)

    def execute(self):
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils import column_or_1d
        # 区分训练或预测，可参照中科院的BDA-studio拆分成两个组件
        if self.mode == "train":
            self.transformedDF = self.originalDF.copy()
            if self.toIntColumns is not None:
                # 转换错误的值不报错，而是以NaN替换，再以defaultIntValue填充缺失值
                for col in self.toIntColumns:
                    self.transformedDF[col] = pd.to_numeric(self.originalDF[col], errors='coerce', downcast='int')\
                        .fillna(self.defaultIntValue)

            if self.toDoubleColumns is not None:
                for col in self.toDoubleColumns:
                    # 转换错误的值不报错，而是以NaN替换，再以defaultDoubleValue填充缺失值
                    self.transformedDF[col] = pd.to_numeric(self.originalDF[col], errors='coerce', downcast='float')\
                        .fillna(self.defaultDoubleValue)

            if self.toCategoricalColumns is not None:
                paramDict = dict()
                for col in self.toCategoricalColumns:
                    le = LabelEncoder()
                    le.fit(self.transformedDF[col])
                    self.transformedDF[col] = le.transform(self.transformedDF[col])
                    # 以#作为标识符记录类型转换的class
                    paramDict[col] = "#".join(map(str, list(le.classes_)))
                self.paramDF = pd.DataFrame(list(paramDict.items()), columns=["col", "classes"])

        elif self.mode == "predict":
            import numpy as np
            # self.paramDF = data.PyReadHive(self.outputUrl2)
            # 读取
            # self.paramDF = self.dataUtil.PyReadCSV(self.outputUrl2)
            self.paramDF = self.dataUtil.PyReadHive(self.outputUrl2)
            self.transformedDF = self.originalDF.copy()
            paramList = self.paramDF.to_dict("records")
            for record in paramList:
                col = record["col"]
                classes_ = record["classes"].split("#")
                y = column_or_1d(self.originalDF[col])
                self.transformedDF[col] = np.array([np.searchsorted(classes_, x) if x in classes_ else -1
                                                    for x in y])

    def setOut(self):
        self.dataUtil.PyWriteHive(self.transformedDF, self.outputUrl1)
        # data.PyWriteHive(self.paramDF, self.outputUrl2)
        # data.PyWriteCSV(self.transformedDF, self.outputUrl1)
        # data.PyWriteCSV(self.paramDF, self.outputUrl2+".csv")


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
    def __init__(self, args, args2):
        """
        Standalone version for type transformation
        @param args: dict
        toDoubleColumns: list
        defaultDoubleValue: double
        toIntColumns: list
        defaultIntValue: int
        toCategoricalColumns: list
        """
        self.originalDF = None
        self.transformedDF = None
        self.paramDF = None

        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = self.outputUrl1+"toCategorical"
        self.param = args["param"]
        try:
            self.toDoubleColumns = self.param["toDouble"]
        except KeyError:
            self.toDoubleColumns = None
        try:
            self.defaultDoubleValue = self.param["defaultDoubleValue"]
        except KeyError:
            self.defaultDoubleValue = 0.0
        try:
            self.toIntColumns = self.param["toInt"]
        except KeyError:
            self.toIntColumns = None
        try:
            self.defaultIntValue = self.param["defaultIntValue"]
        except KeyError:
            self.defaultIntValue = 0
        try:
            self.toCategoricalColumns = self.param["toCategoricalColumns"]
        except KeyError:
            self.toCategoricalColumns = None

        self.mode = self.param["mode"]

        print("using PySpark")

        self.spark = utils.init_spark()

    def getIn(self):
        self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl1, self.spark)

    def execute(self):
        from pyspark.sql.functions import udf

        # 区分训练或预测，可参照中科院的BDA-studio拆分成两个组件
        if self.mode == "train":
            self.transformedDF = self.originalDF
            if self.toIntColumns is not None:
                # 定义UDF
                def to_int(x):
                    try:
                        x = int(x)
                    except ValueError as e:
                        x = int(self.defaultIntValue)
                    return x

                for col in self.toIntColumns:
                    self.transformedDF = self.transformedDF.withColumn(col, udf(to_int)(col))

            if self.toDoubleColumns is not None:
                # 定义UDF
                def to_double(x):
                    try:
                        x = float(x)
                    except ValueError as e:
                        x = float(self.defaultDoubleValue)
                    return x
                for col in self.toDoubleColumns:
                    self.transformedDF = self.transformedDF.withColumn(col, udf(to_double)(col))

            if self.toCategoricalColumns is not None:
                from pyspark.ml.feature import StringIndexer
                indexers = [StringIndexer(inputCol=column, outputCol="index_" + column).fit(self.originalDF) for column
                            in self.toCategoricalColumns]
                self.transformedDF = self.originalDF
                for i in range(len(indexers)):
                    self.transformedDF = indexers[i].transform(self.transformedDF).drop(
                        self.toCategoricalColumns[i]).withColumnRenamed("index_" + self.toCategoricalColumns[i],
                                                                        self.toCategoricalColumns[i])

    def setOut(self):
        utils.dataUtil.SparkWriteHive(self.transformedDF, self.outputUrl1)


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
