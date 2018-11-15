from pyhive import hive
import os
import xml.etree.ElementTree as ET
def get_connection():
    username_and_password = _get_username_and_password()
    conn = hive.Connection(host='127.0.0.1', port='10000', username=username_and_password[0], password=username_and_password[1],
                           auth='CUSTOM')
    return conn

def get_cursor():
    return get_connection().cursor()

def _get_username_and_password():
    root = ET.parse(os.getenv("HIVE_HOME") + "/conf/hive-site.xml").getroot()
    return [[i[1].text for i in root.iter(tag="property") if i[0].text=="javax.jdo.option.ConnectionUserName"][0], \
        [i[1].text for i in root.iter(tag="property") if i[0].text=="javax.jdo.option.ConnectionPassword"][0]]

def mapping_df_types(df):
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
            dtypedict.update({i :"BOOLEAN"})
    return dtypedict
