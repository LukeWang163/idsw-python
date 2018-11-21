#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.data.data.py
# @Desc    : Scripts for reading and writing data from/to multiple sources.
from .. import utils


class dataUtil:
    def __init__(self, args):
        self.args = args

    def PyReadHive(self, inputUrl):
        """
        Standalone version for reading data from Hive
        @param inputUrl: String
        @return: Pandas.DataFrame
        """
        import pandas as pd
        conn = utils.get_connection(self.args)
        return pd.read_sql("select * from %s" % inputUrl, conn)

    def PyWriteHive(self, df, outputUrl):
        """
        Standalone version for writing dataframe to Hive
        @param df: Pandas.DataFrame
        @param outputUrl: String
        @return:
        """
        import os
        # 映射获得表结构
        dtypeDict = utils.mapping_df_types(df)
        dtypeString = ",".join("`{}` {}".format(*i) for i in dtypeDict.items())
        # 获得Hive连接
        cursor = utils.get_cursor(self.args)
        # 建表
        cursor.execute("drop table if exists %s" % outputUrl)
        cursor.execute("create table %s (%s) row format delimited fields terminated by '\t'" % (outputUrl, dtypeString))
        # 将数据写到本地临时txt文件
        df.to_csv("/tmp/" + outputUrl + ".txt", header=False, index=False, sep="\t")
        # 将本地临时txt文件内容插入表
        cursor.execute("load data local inpath '%s' overwrite into table %s" % ("/tmp/" + outputUrl + ".txt", outputUrl))
        # 删除本地临时txt文件
        os.remove("/tmp/" + outputUrl + ".txt")
        print("writen to Hive")
        return

    def PyReadCSV(self, inputUrl):
        """
        Standalone version for reading from CSV file
        @param inputUrl: String
        @return: Pandas.DataFrame
        """
        import pandas as pd
        return pd.read_csv(inputUrl, encoding='utf-8')

    def PyWriteCSV(self, df, outputUrl):
        """
        Standalone version for writing dataframe to CSV file
        @param df: Pandas.DataFrame
        @param outputUrl: String
        @return:
        """
        df.to_csv(outputUrl, index=False, encoding='utf-8')
        return

    def SparkReadHive(self, inputUrl, spark):
        """
        Spark version for reading from Hive
        @param inputUrl: String
        @param spark: SparkSession
        @return: pyspark.sql.DataFrame
        """
        return spark.sql("select * from " + inputUrl)

    def SparkWriteHive(self, DF, outputUrl):
        """
        Spark version for writing to Hive
        @param DF: pyspark.sql.DataFrame
        @param outputUrl: String
        @return:
        """
        DF.write.mode("overwrite").format("hive").saveAsTable(outputUrl)
        return

    def SparkReadCSV(self, inputUrl, spark):
        """
        Spark version for reading from CSV file
        @param inputUrl: String
        @param spark: SparkSession
        @return: pyspark.sql.DataFrame
        """
        return spark.read\
            .option("header", True).option("inferSchema", True).option("mode", "DROPMALFORMED")\
            .format("csv")\
            .load(inputUrl)
