#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.utils.py
# @Desc    : Utils for idsw modules, including establishing Hive connection, map dataframe dtyps to hive dtypes, etc.
from pyhive import hive
import json


def get_connection(args):
    """
    Get connection with Hive
    @return: pyhive.Connection
    """
    username_and_password = _get_username_and_password(args)
    conn = hive.Connection(host='127.0.0.1', port='10000',
                           username=username_and_password[0],
                           password=username_and_password[1],
                           auth='CUSTOM', configuration={"hive.resultset.use.unique.column.names": "false"})
    return conn


def get_cursor(args):
    """
    Get cursor with Hive
    @return: pyhive.Connection.cursor
    """
    return get_connection(args).cursor()


def _get_username_and_password(args):
    """
    Parse hive-site.xml, then get username and password
    @return: list
    """
    args = json.loads(args.replace("\!", "!"))
    return [args["username"], args["password"]]
    #root = ET.parse(os.getenv("HIVE_HOME") + "/conf/hive-site.xml").getroot()
    #return [[i[1].text for i in root.iter(tag="property") if i[0].text == "javax.jdo.option.ConnectionUserName"][0],
    #        [i[1].text for i in root.iter(tag="property") if i[0].text == "javax.jdo.option.ConnectionPassword"][0]]


def mapping_df_types(df):
    """
    Mapping table for transformation from dataframe dtype to Hive table dtype
    @param df: Pandas.DataFrame
    @return: dict
    """
    from collections import OrderedDict
    dtypedict = OrderedDict()
    for i, j in zip(df.columns, df.dtypes):
        if "object" in str(j):
            dtypedict.update({i: "STRING"})
        if "float" in str(j):
            dtypedict.update({i: "FLOAT"})
        if "int" in str(j):
            dtypedict.update({i: "INT"})
        if "datetime" in str(j):
            dtypedict.update({i: "TIMESTAMP"})
        if "bool" in str(j):
            dtypedict.update({i: "BOOLEAN"})
    return dtypedict


def init_spark():
    from pyspark.sql import SparkSession
    return SparkSession \
        .builder \
        .master("yarn") \
        .enableHiveSupport() \
        .getOrCreate()
