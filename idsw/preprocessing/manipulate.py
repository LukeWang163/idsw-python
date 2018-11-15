from ..data import data


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

        self.originalDF = None
        self.transformedDF = None
        self.paramDF = None

        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = self.outputUrl1+"toCategorical"
        self.param = args["param"]
        try:
            self.toDoubleColumns = self.param["toDouble"].split(",")
        except KeyError:
            self.toDoubleColumns = None
        try:
            self.defaultDoubleValue = self.param["defaultDoubleValue"]
        except KeyError:
            self.defaultDoubleValue = 0.0
        try:
            self.toIntColumns = self.param["toInt"].split(",")
        except KeyError:
            self.toIntColumns = None
        try:
            self.defaultIntValue = self.param["defaultIntValue"]
        except KeyError:
            self.defaultDoubleValue = 0.0
        try:
            self.toCategoricalColumns = self.param["toCategoricalColumns"].split(",")
        except KeyError:
            self.toCategoricalColumns = None

        self.mode = (self.param["model"])

    def getIn(self):

        # self.originalDF = data.PyReadHive(self.inputUrl1)
        self.originalDF = data.PyReadCSV(self.inputUrl1)

    def execute(self):
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils import column_or_1d
        if self.mode == "train":
            self.transformedDF = self.originalDF.copy()
            if self.toIntColumns is not None:
                for col in self.toIntColumns:
                    self.transformedDF[col] = pd.to_numeric(self.originalDF[col], errors='coerce', downcast='int')\
                        .fillna(self.defaultIntValue)

            if self.toDoubleColumns is not None:
                for col in self.toDoubleColumns:
                    self.transformedDF[col] = pd.to_numeric(self.originalDF[col], errors='coerce', downcast='float')\
                        .fillna(self.defaultDoubleValue)

            if self.toCategoricalColumns is not None:
                paramDict = dict()
                for col in self.toCategoricalColumns:
                    le = LabelEncoder()
                    le.fit(self.transformedDF[col])
                    self.transformedDF[col] = le.transform(self.transformedDF[col])
                    paramDict[col] = "#".join(list(le.classes_))
                self.paramDF = pd.DataFrame(list(paramDict.items()), columns=["col", "classes"])

        elif self.mode == "predict":
            import numpy as np
            # self.paramDF = data.PyReadHive(self.outputUrl2)
            self.paramDF = data.PyReadCSV(self.outputUrl2)
            self.transformedDF = self.originalDF.copy()
            paramList = self.paramDF.to_dict("records")
            for record in paramList:
                col = record["col"]
                classes_ = record["classes"].split("#")
                y = column_or_1d(self.originalDF[col])
                self.transformedDF[col] = np.array([np.searchsorted(classes_, x) if x in classes_ else -1
                                                    for x in y])


    def setOut(self):
        # data.PyWriteHive(self.transformedDF, self.outputUrl1)
        # data.PyWriteHive(self.paramDF, self.outputUrl2)
        data.PyWriteCSV(self.transformedDF, self.outputUrl1)
        data.PyWriteCSV(self.paramDF, self.outputUrl2+".csv")

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
