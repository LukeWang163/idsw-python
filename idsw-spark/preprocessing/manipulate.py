#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.preprocessing.manipulate.py
# @Desc    : Scripts for manipulating data. 数据预处理->数据操作
import utils


class Join:
    pass


class kMergeRow:
    pass


class MergeColumn:
    pass


class SelectColumn:
    pass


class GroupBy:
    pass


class TypeTransform:
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


class Replace:
    pass


class AsColumn:
    pass


class Transpose:
    pass


class Table2KV:
    pass


class KV2Table:
    pass


class HandleMissingValue:
    pass


class ChangeColumnName:
    pass


class AddId:
    pass


class DropDuplicate:
    pass


class SortValue:
    pass
