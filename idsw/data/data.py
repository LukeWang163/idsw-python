def PyReadHive(inputUrl):
    from .. import utils
    import pandas as pd
    cursor = utils.get_cursor()
    return pd.read_sql("select * from %s"%(inputUrl1))

def PyWriteHive(df, outputUrl1):
    from .. import utils
    import pandas as pd
    import os
    dtypedict = utils.mapping_df_types(df)
    dtypeString = ",".join("{} {}".format(*i) for i in dtypedict.items())
    cursor.execute("drop table if exists %s"%(outputUrl1))
    cursor.execute("create table %s (%s) row format delimited fields terminated by '\t'"%(outputUrl1, dtypeString))
    df.to_csv("/tmp/" + outputUrl1 + ".txt", header=False, index=False, sep="\t")
    cursor.execute("load data local inpath '%s' overwrite into table %s" %("/tmp/" + outputUrl1 + ".txt", outputUrl1))
    os.remove("/tmp/" + outputUrl1 + ".txt")
    print("writen to Hive")
    return

def PyReadCSV(inputUrl):

    import pandas as pd
    return pd.read_csv(inputUrl, encoding='utf-8')

def SparkReadCSV(inputUrl, spark):

    return spark.read.option("header", True).option("inferSchema", True).option("mode", "DROPMALFORMED").format("csv").load(inputUrl);
