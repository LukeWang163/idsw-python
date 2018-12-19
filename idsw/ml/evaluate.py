#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.ml.evaluate.py
# @Desc    : Evaluation scripts for our built models. 机器学习->评估
import utils
import logging


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
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = args["output"][1]["value"]
        self.labelCol = args["param"]["label"]
        self.posLabel = args["param"]["posLabel"]
        self.originalDF = None
        self.metric_df = None
        self.roc_pr = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        self.logger.debug("loading input data")
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl1)
        # self.originalDF = data.PyReadCSV(self.inputUrl1)

    def execute(self):
        # 评估指标表
        import pandas as pd
        import numpy as np
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
        y_true = self.originalDF[self.labelCol]
        y_pred = self.originalDF["prediction"]
        if self.posLabel not in y_true:
            if int(self.posLabel) in y_true:
                self.posLabel = int(self.posLabel)
            elif float(self.posLabel) in y_true:
                self.posLabel = float(self.labelCol)
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, pos_label=self.posLabel)
            recall = recall_score(y_true, y_pred, pos_label=self.posLabel)
            f1 = f1_score(y_true, y_pred, pos_label=self.posLabel)
            auc = roc_auc_score(y_true.apply(lambda x: 1 if str(x) == str(self.posLabel) else 0),
                                self.originalDF["predicted as '%s'" % (str(self.posLabel))])
            logloss = log_loss(y_true, self.originalDF["predicted as '%s'" % (str(self.posLabel))])
        except ValueError:
            self.logger.error("Not a valid pos_label")
            import sys
            sys.exit(0)

        metric_dict = {"accuracy": accuracy, "recall": recall, "precision": precision, "f1 score": f1, "auc": auc,
                       "logloss": logloss}
        cm_dict = dict(
            zip(["TN", "FP", "FN", "TP"], confusion_matrix(self.originalDF[self.labelCol], self.originalDF["prediction"]).ravel()))
        metric_dict.update(cm_dict)

        self.metric_df = pd.DataFrame(metric_dict, index=[0])

        # ROC、PR表
        T = self.originalDF["predicted as '%s'" % (str(self.posLabel))]
        Y = self.originalDF[self.labelCol].apply(lambda x: 1 if str(x) == str(self.posLabel) else 0)
        thresholds = np.linspace(1, 0, 101)
        ROC = np.zeros((101, 2))
        PR = np.zeros((101, 2))

        for i in range(101):
            t = thresholds[i]
            # Classifier / label agree and disagreements for current threshold.
            TP_t = np.logical_and(T > t, Y == 1).sum()
            TN_t = np.logical_and(T <= t, Y == 0).sum()
            FP_t = np.logical_and(T > t, Y == 0).sum()
            FN_t = np.logical_and(T <= t, Y == 1).sum()

            # Compute false positive rate for current threshold.
            FPR_t = FP_t / float(FP_t + TN_t)
            ROC[i, 0] = FPR_t

            # Compute true  positive rate for current threshold.
            TPR_t = TP_t / float(TP_t + FN_t)
            ROC[i, 1] = TPR_t

            # Compute false positive rate for current threshold.
            P_t = TP_t / float(TP_t + FP_t)
            PR[i, 0] = P_t

            # Compute true  positive rate for current threshold.
            R_t = TP_t / float(TP_t + FN_t)
            PR[i, 1] = R_t

        self.roc_pr = pd.DataFrame(
            {"threshold": thresholds, "FPR": ROC[:, 0], "TPR": ROC[:, 1], "P": PR[:, 0], "R": PR[:, 1]})

    def setOut(self):
        self.logger.info("saving evaluation result to %s" % self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.metric_df, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.roc_pr, self.outputUrl2)


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
        self.outputUrl1 = args["output"][0]["value"]
        self.labelCol = args["param"]["label"]
        self.posLabel = args["param"]["posLabel"]
        self.originalDF = None
        self.metric_df = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        self.logger.debug("loading input data")
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl1)
        # self.originalDF = data.PyReadCSV(self.inputUrl1)

    def execute(self):
        import pandas as pd
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        y_true = self.originalDF[self.labelCol]
        y_pred = self.originalDF["prediction"]
        accuracy = accuracy_score(y_true, y_pred)
        precision_micro = precision_score(y_true, y_pred, average="micro")
        recall_micro = recall_score(y_true, y_pred, average="micro")
        f1_micro = f1_score(y_true, y_pred, average="micro")
        precision_macro = precision_score(y_true, y_pred, average="macro")
        recall_macro = recall_score(y_true, y_pred, average="macro")
        f1_macro = f1_score(y_true, y_pred, average="macro")

        labelList = [col.split("'")[1] for col in self.originalDF.columns if "predicted as " in col]
        unique_label = y_true.unique()
        label_list = [
            label if label in unique_label else float(label) if float(label) in unique_label else int(label) if int(
                label) in unique_label else None for label in labelList]
        cm = confusion_matrix(y_true, y_pred, labels=labelList)
        cm_dict = dict([(labelList[i], cm[:, i]) for i in range(len(labelList))])

        metric_dict = {"accuracy": accuracy, "micro-precision": precision_micro, "micro-recall": recall_micro,
                       "micro-f1": f1_micro, "macro-precision": precision_macro, "macro-recall": recall_macro,
                       "macro_f1": f1_macro}
        metric_dict.update(cm_dict)
        self.metric_df = pd.DataFrame(metric_dict)

    def setOut(self):
        self.logger.info("saving evaluation result to %s" % self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.metric_df, self.outputUrl1)


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
        self.outputUrl1 = args["output"][0]["value"]
        self.labelCol = args["param"]["label"]
        self.originalDF = None
        self.metric_df = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        self.logger.debug("loading input data")
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl1)
        # self.originalDF = utils.dataUtil.PyReadCSV(self.inputUrl2)

    def execute(self):
        from collections import OrderedDict
        import pandas as pd
        import numpy as np
        from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
        y_true = self.originalDF[self.labelCol]
        y_pred = self.originalDF["prediction"]
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = median_absolute_error(y_true, y_pred)
        SSE = ((y_true - y_pred) ** 2).sum(axis=0)
        SST = ((y_true - np.average(y_pred, axis=0)) ** 2).sum(axis=0)
        SSR = SST - SSE

        self.metric_df = pd.DataFrame.from_dict(
            OrderedDict({"r2": [r2], "rmse": [rmse], "mae": [mae], "SSE": [SSE], "SST": [SST], "SSR": [SSR]}))

    def setOut(self):
        self.logger.info("saving evaluation result to %s" % self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.metric_df, self.outputUrl1)


class EvaluateClustering:
    def __init__(self, args, args2):
        """
        Standalone version for evaluating clustering model
        @param args: dict
        featureCols: list
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.featureCols = args["param"]["features"]

        self.originalDF = None
        self.metric_df = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # sklearn等模型加载
        self.logger.debug("using scikit-learn")
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl1)
        # self.originalDF = data.PyReadCSV(self.inputUrl2)

    def execute(self):
        from collections import OrderedDict
        # sklearn等模型评估
        import pandas as pd
        from sklearn.metrics import silhouette_score
        self.logger.info("evaluating model")
        silhouetteScore = silhouette_score(self.originalDF[self.featureCols], self.originalDF["prediction"])

        self.logger.info("r2: %s" % silhouetteScore)

        self.metric_df = pd.DataFrame.from_dict(
            OrderedDict({"accuracy": [silhouetteScore]}))

    def setOut(self):
        self.logger.info("saving evaluation result to %s" % self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.metric_df, self.outputUrl1)
