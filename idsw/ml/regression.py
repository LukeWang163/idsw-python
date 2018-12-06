#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.ml.regression.py
# @Desc    : Scripts for initializing regression models. 机器学习->算法选择->回归
import utils
import logging


class LinearRegression:
    pass


class GLM:
    pass


class GBDT:
    pass


class RandomForest:
    def __init__(self, args, args2):
        """
        Standalone version for initializing RandomForest regressor
        @param args: dict
        n_estimators: int
        criterion: string one of "mse" and "mae"
        max_depth: int
        min_samples_split: int
        min_samples_leaf: int
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]
        self.model = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        return

    def execute(self):
        self.logger.debug("using scikit-learn")
        # 树的个数
        n_estimators = int(self.param["treeNum"])
        # 评价标准
        criterion = self.param["criterion"]
        # 最大树深度
        max_depth = int(self.param["maxDepth"])
        # 最小分割样本数
        min_samples_split = int(self.param["minSamplesSplit"])
        # 叶子节点最小样本数
        min_samples_leaf = int(self.param["minSamplesLeaf"])

        # 初始化模型
        self.logger.info("initializing model")
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    def setOut(self):
        self.logger.info("saving model to %s" % self.outputUrl1)
        self.dataUtil.PyWriteModel(self.model, self.outputUrl1)


class PossionRegression:
    pass


class CoxRegression:
    pass


class MLP:
    pass


class AutoML:
    pass
