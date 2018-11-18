def PyReadHive(inputUrl):
    """
    Standalone version for reading data from Hive
    @param inputUrl: String
    @return: Pandas.DataFrame
    """
    from .. import utils
    import pandas as pd
    conn = utils.get_connection()
    return pd.read_sql("select * from %s" % inputUrl, conn)


def PyWriteHive(df, outputUrl):
    """
    Standalone version for writing dataframe to Hive
    @param df: Pandas.DataFrame
    @param outputUrl: String
    @return:
    """
    from .. import utils
    import os
    dtypeDict = utils.mapping_df_types(df)
    dtypeString = ",".join("{} {}".format(*i) for i in dtypeDict.items())
    cursor = utils.get_cursor()
    cursor.execute("drop table if exists %s" % outputUrl)
    cursor.execute("create table %s (%s) row format delimited fields terminated by '\t'" % (outputUrl, dtypeString))
    df.to_csv("/tmp/" + outputUrl + ".txt", header=False, index=False, sep="\t")
    cursor.execute("load data local inpath '%s' overwrite into table %s" % ("/tmp/" + outputUrl + ".txt", outputUrl))
    os.remove("/tmp/" + outputUrl + ".txt")
    print("writen to Hive")
    return


def PyReadCSV(inputUrl):
    """
    Standalone version for reading from CSV file
    @param inputUrl: String
    @return: Pandas.DataFrame
    """
    import pandas as pd
    return pd.read_csv(inputUrl, encoding='utf-8')


def PyWriteCSV(df, outputUrl):
    """
    Standalone version for writing dataframe to CSV file
    @param df: Pandas.DataFrame
    @param outputUrl: String
    @return:
    """
    df.to_csv(outputUrl, index=False, encoding='utf-8')
    return


def SparkReadHive(inputUrl, spark):
    """
    Spark version for reading from Hive
    @param inputUrl: String
    @param spark: SparkSession
    @return: pyspark.sql.DataFrame
    """
    return spark.sql("select * from " + inputUrl)


def SparkWriteHive(DF, outputUrl):
    """
    Spark version for writing to Hive
    @param DF: pyspark.sql.DataFrame
    @param outputUrl: String
    @return:
    """
    DF.write.mode("overwrite").format("hive").saveAsTable(outputUrl)
    return


def SparkReadCSV(inputUrl, spark):
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