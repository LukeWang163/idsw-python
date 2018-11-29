#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.ml.evaluate.py
# @Desc    : Evaluation scripts for our built models. 机器学习->评估
import utils
import logging
import logging.config
logging.config.fileConfig('logging.ini')


class CrossValidate:
    pass


class EvaluateBinaryClassifier:
    def __init__(self, args, args2):
        """
        Standalone version for evaluating binary classifier
        @param args: dict
        featureCols: list
        labelCol: string
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]

        self.originalDF = None
        self.transformDF = None
        self.model = None
        self.result = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # sklearn等模型加载
        self.logger.debug("using scikit-learn")
        import pandas as pd
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
        # self.originalDF = data.PyReadCSV(self.inputUrl2)

        self.model = self.dataUtil.PyReadModel(self.inputUrl1)

    def execute(self):
        from collections import OrderedDict
        featureCols = self.param["features"]
        labelCol = self.param["label"]

        # sklearn等模型评估
        import pandas as pd
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        self.transformDF = self.originalDF.copy()
        self.logger.info("evaluating model")
        predicted = self.model.predict(self.originalDF[featureCols])

        self.transformDF["prediction"] = predicted

        accuracy = accuracy_score(self.originalDF[labelCol], predicted)
        precision = precision_score(self.originalDF[labelCol], predicted)
        recall = recall_score(self.originalDF[labelCol], predicted)
        f1 = f1_score(self.originalDF[labelCol], predicted)

        self.result = pd.DataFrame.from_dict(
            OrderedDict({"accuracy": [accuracy], "precision": [precision], "recall": [recall], "f1 score": [f1]}))

    def setOut(self):
        self.logger.info("saving evaluation result to %s" % self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.result, self.outputUrl1)


class EvaluateMultiClassClassifier:
    def __init__(self, args, args2):
        """
        Standalone version for evaluating multi-class classifier
        @param args: dict
        featureCols: list
        labelCol: string
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]

        self.originalDF = None
        self.transformDF = None
        self.model = None
        self.result = None
        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # sklearn等模型加载
        self.logger.debug("using scikit-learn")
        import pandas as pd
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
        # self.originalDF = data.PyReadCSV(self.inputUrl2)

        self.model = self.dataUtil.PyReadModel(self.inputUrl1)

    def execute(self):
        from collections import OrderedDict
        featureCols = self.param["features"]
        labelCol = self.param["label"]

        # sklearn等模型评估
        import pandas as pd
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        self.transformDF = self.originalDF.copy()
        self.logger.info("evaluating model")
        predicted = self.model.predict(self.originalDF[featureCols])

        self.transformDF["prediction"] = predicted

        accuracy = accuracy_score(self.originalDF[labelCol], predicted)
        precision = precision_score(self.originalDF[labelCol], predicted)
        recall = recall_score(self.originalDF[labelCol], predicted)
        f1 = f1_score(self.originalDF[labelCol], predicted)

        self.result = pd.DataFrame.from_dict(
            OrderedDict({"accuracy": [accuracy], "precision": [precision], "recall": [recall], "f1 score": [f1]}))

    def setOut(self):
        self.logger.info("saving evaluation result to %s" % self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.result, self.outputUrl1)


class EvaluateRegressor:
    def __init__(self, args, args2):
        """
        Standalone version for evaluating regressor
        @param args: dict
        featureCols: list
        labelCol: string
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]

        self.originalDF = None
        self.transformDF = None
        self.model = None
        self.result = None
        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # sklearn等模型评估
        self.logger.debug("using scikit-learn")
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
        # self.originalDF = utils.dataUtil.PyReadCSV(self.inputUrl2)
        self.model = self.dataUtil.PyReadModel(self.inputUrl1)

    def execute(self):
        from collections import OrderedDict
        featureCols = self.param["features"]
        labelCol = self.param["label"]

        import pandas as pd
        import numpy as np
        from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
        self.logger.info("evaluating model")
        self.transformDF = self.originalDF.copy()
        predicted = self.model.predict(self.originalDF[featureCols])

        self.transformDF["prediction"] = predicted

        r2 = r2_score(self.originalDF[labelCol], predicted)
        rmse = np.sqrt(mean_squared_error(self.originalDF[labelCol], predicted))
        mae = median_absolute_error(self.originalDF[labelCol], predicted)

        self.result = pd.DataFrame.from_dict(
            OrderedDict({"r2": [r2], "rmse": [rmse], "mae": [mae]}))

    def setOut(self):
        self.logger.info("saving evaluation result to %s" % self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.result, self.outputUrl1)


class EvaluateClustering:
    def __init__(self, args, args2):
        """
        Standalone version for evaluating clustering model
        @param args: dict
        featureCols: list
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]

        self.originalDF = None
        self.model = None
        self.result = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # sklearn等模型加载
        self.logger.debug("using scikit-learn")
        import pandas as pd
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
        # self.originalDF = data.PyReadCSV(self.inputUrl2)
        self.model = self.dataUtil.PyReadModel(self.inputUrl1)

    def execute(self):
        from collections import OrderedDict
        featureCols = self.param["features"]
        # sklearn等模型评估
        import pandas as pd
        from sklearn.metrics import silhouette_score
        self.logger.info("evaluating model")
        silhouetteScore = silhouette_score(self.originalDF[featureCols], self.originalDF["prediction"])

        self.logger.info("r2: %s" % silhouetteScore)

        self.result = pd.DataFrame.from_dict(
            OrderedDict({"accuracy": [silhouetteScore]}))

    def setOut(self):
        self.logger.info("saving evaluation result to %s" %self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.result, self.outputUrl1)

