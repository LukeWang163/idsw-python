#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 12/19/18 10:24 AM
# @Author  : Luke
# @File    : automl.py
# @Desc    : Script for initializing and training auto classification model.
import logging
import logging.config
import utils
import time


class AutoClassification:
    def __init__(self, args, args2):
        """
        Standalone version for initializing and training auto-sklearn model
        @param args: dict
        time_limit: int
        time_per_model: int
        ensemble_size: int
        metric: string one of "roc_auc", "accuracy", "precision" and "f1"
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]
        self.originalDF = None
        self.model = None
        self.dataUtil = utils.dataUtil(args2)
        current_time = str(time.time())
        self.tmp_folder = 'tmp/auto_classification_' + current_time + "_tmp"
        self.out_folder = 'tmp/auto_classification_' + current_time + "_out"

    def getIn(self):
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl1)

    def execute(self):
        import autosklearn.classification
        from autosklearn.metrics import roc_auc, accuracy, precision, f1
        # 特征列
        featureCols = self.param["features"]
        # 标签列
        labelCol = self.param["label"]
        # 训练时间限制
        time_limit = int(self.param["timeLimit"])
        # 每个模型训练时间限制
        time_per_model = int(self.param["timePer"])
        # 最多保留模型个数
        ensemble_size = int(self.param["ensembleSize"])
        # 评价指标
        metric = self.param["metric"]
        assert time_per_model < time_limit, "time_per_model should be less than time_limit"

        self.logger.info("initializing model")

        self.model = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=time_limit,
            per_run_time_limit=time_per_model,
            tmp_folder=self.tmp_folder,
            output_folder=self.out_folder,
            ensemble_size=ensemble_size,
            ml_memory_limit=3072,
            delete_output_folder_after_terminate=False,
            delete_tmp_folder_after_terminate=False,
            shared_mode=True
        )

        metric_map = {"roc_auc": roc_auc, "accuracy": accuracy, "precision": precision, "f1": f1}
        self.model.fit(self.originalDF[featureCols], self.originalDF[labelCol], metric=metric_map[metric])

    def setOut(self):
        self.logger.info("saving trained auto-ml model to %s" % self.outputUrl1)
        self.dataUtil.PyWriteModel(self.model, self.outputUrl1)
        import shutil
        try:
            shutil.rmtree(self.out_folder)
            shutil.rmtree(self.tmp_folder)
        except Exception:
            self.logger.warning("Could not delete output dir: %s" % self.out_folder)


class AutoRegression:
    def __init__(self, args, args2):
        """
        Standalone version for initializing and training auto-sklearn model
        @param args: dict
        time_limit: int
        time_per_model: int
        ensemble_size: int
        metric: string one of "roc_auc", "accuracy", "precision" and "f1"
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]
        self.originalDF = None
        self.model = None
        self.dataUtil = utils.dataUtil(args2)
        current_time = str(time.time())
        self.tmp_folder = 'tmp/auto_regression_' + current_time + "_tmp"
        self.out_folder = 'tmp/auto_regression_' + current_time + "_out"

    def getIn(self):
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl1)

    def execute(self):
        import autosklearn.regression
        from autosklearn.metrics import r2, mean_squared_error, median_absolute_error
        # 特征列
        featureCols = self.param["features"]
        # 标签列
        labelCol = self.param["label"]
        # 训练时间限制
        time_limit = int(self.param["timeLimit"])
        # 每个模型训练时间限制
        time_per_model = int(self.param["timePer"])
        # 最多保留模型个数
        ensemble_size = int(self.param["ensembleSize"])
        # 评价指标
        metric = self.param["metric"]
        assert time_per_model < time_limit, "time_per_model should be less than time_limit"

        self.logger.info("initializing model")

        self.model = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=time_limit,
            per_run_time_limit=time_per_model,
            tmp_folder=self.tmp_folder,
            output_folder=self.out_folder,
            ensemble_size=ensemble_size,
            ml_memory_limit=3072,
            delete_output_folder_after_terminate=False,
            delete_tmp_folder_after_terminate=False,
            shared_mode=True
        )

        metric_map = {"r2": r2, "mse": mean_squared_error, "mae": median_absolute_error}
        self.model.fit(self.originalDF[featureCols], self.originalDF[labelCol], metric=metric_map[metric])

    def setOut(self):
        self.logger.info("saving trained auto-ml model to %s" % self.outputUrl1)
        self.dataUtil.PyWriteModel(self.model, self.outputUrl1)
        import shutil
        try:
            shutil.rmtree(self.out_folder)
            shutil.rmtree(self.tmp_folder)
        except Exception:
            self.logger.warning("Could not delete output dir: %s" % self.out_folder)


class Predict:
    def __init__(self, args, args2):
        """
        Standalone version for predicting auto classification and regression algo
        @param args: dict
        featureCols: list
        labelCol: string
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.featureCols = args["param"]["features"]

        self.originalDF = None
        self.transformDF = None
        self.model = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        # sklearn等模型加载
        self.logger.info("using auto-sklearn")
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl2)
        self.model = self.dataUtil.PyReadModel(self.inputUrl1)

    def execute(self):
        import sys
        # sklearn等模型评估
        self.transformDF = self.originalDF.copy()
        self.logger.info(f"predicting {str(self.model.__class__)} model")
        # judge type
        modelType = None
        print(self.model.target_type)
        try:
            if self.model.target_type == "binary":
                labelList = self.model._automl._classes[0]
                if len(labelList) == 2:
                    self.logger.info("predicting binary classification model")
                    modelType = "Binary"
                else:
                    self.logger.error("model classes number is wrong. double check")
                    sys.exit(0)
            elif self.model.target_type == "multiclass":
                labelList = self.model._automl._classes[0]
                if len(labelList) > 2:
                    self.logger.info("predicting multi-class classification model")
                    modelType = "Multi"
                else:
                    self.logger.error("model classes number is wrong. double check")
                    sys.exit(0)
        except AttributeError:
            if str(self.model.__class__) == "<class 'autosklearn.estimators.AutoSklearnRegressor'>":
                self.logger.info("predicting regression model")
                modelType = "Reg"
            else:
                self.logger.error("unsupported model type")
                sys.exit(0)

        def executeBinary():
            predicted = self.model.predict(self.originalDF[self.featureCols])
            predicted_proba = self.model.predict_proba(self.originalDF[self.featureCols])
            self.transformDF["prediction"] = predicted
            for i in range(len(labelList)):
                self.transformDF["predicted as '%s'" % str(labelList[i])] = predicted_proba[:, i]

        def executeMulti():
            predicted = self.model.predict(self.originalDF[self.featureCols])
            predicted_proba = self.model.predict_proba(self.originalDF[self.featureCols])
            self.transformDF["prediction"] = predicted
            for i in range(len(labelList)):
                self.transformDF["predicted as '%s'" % str(labelList[i])] = predicted_proba[:, i]

        def executeReg():
            predicted = self.model.predict(self.originalDF[self.featureCols])
            self.transformDF["prediction"] = predicted

        # 三类模型
        modelTypes = {"Binary": executeBinary, "Multi": executeMulti, "Reg": executeReg}
        if modelType is not None:
            modelTypes[modelType]()
        else:
            self.logger.error("not supported")
            import sys
            sys.exit(0)

    def setOut(self):
        self.logger.info("saving predicting result to %s" % self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.transformDF, self.outputUrl1)


class EvaluateAutoClassifier:
    def __init__(self, args, args2):
        """
        Standalone version for evaluating auto classifier
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
        self.logger.debug("loading predicted result")
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl1)

    def execute(self):
        import pandas as pd
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        y_true = self.originalDF[self.labelCol]
        y_pred = self.originalDF["prediction"]
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average="macro")
        recall_macro = recall_score(y_true, y_pred, average="macro")
        f1_macro = f1_score(y_true, y_pred, average="macro")

        labelList = [col.split("'")[1] for col in self.originalDF.columns if "predicted as " in col]
        unique_label = y_true.unique()
        label_list = [label if label in unique_label else float(label) if float(label) in unique_label else int(label) if int(label) in unique_label else None for label in labelList]
        cm = confusion_matrix(y_true, y_pred, labels=label_list)
        cm_dict = dict([(labelList[i], cm[:, i]) for i in range(len(labelList))])

        metric_dict = {"accuracy": accuracy, "macro-precision": precision_macro, "macro-recall": recall_macro,
                       "macro_f1": f1_macro}
        metric_dict.update(cm_dict)
        self.metric_df = pd.DataFrame(metric_dict)

    def setOut(self):
        self.logger.info("saving evaluation result to %s" % self.outputUrl1)
        # data.PyWriteCSV(self.result, self.outputUrl1)
        self.dataUtil.PyWriteHive(self.metric_df, self.outputUrl1)


class EvaluateAutoRegressor:
    def __init__(self, args, args2):
        """
        Standalone version for evaluating auto regressor
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
        self.dataUtil.PyWriteHive(self.metric_df, self.outputUrl1)
