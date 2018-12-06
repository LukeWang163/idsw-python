#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.ml.binary.py
# @Desc    : Scripts for initializing binary classification models. 机器学习->模型初始化->二分类
import logging
import utils


class SVM:
    pass


class DecisionTree:
    pass


class RandomForest:
    def __init__(self, args, args2):
        """
        Standalone version for initializing RandomForest binary classifier
        @param args: dict
        n_estimators: int
        criterion: string one of "gini" and "entropy"
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
        self.logger.info("using scikit-learn")
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
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    def setOut(self):
        self.logger.info("saving initialized model to %s" % self.outputUrl1)
        self.dataUtil.PyWriteModel(self.model, self.outputUrl1)


class GBDT:
    pass


class LogisticRegression:
    pass


class NaiveBayes:
    pass


class BayesNetwork:
    pass


class AveragedPerceptron:
    pass


class MLP:
    pass


class AutoML:
    pass
