#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.featuring.transform.py
# @Desc    : Scripts for feature transformation. 特征工程->特征变换
import utils
import logging

class PCA:
    def __init__(self, args, args2):
        """
        Python version for conducting PCA on input dataset
        @param args: dict
        inputUrl1: String
        outputUrl1: String
        columns: list
        k: int
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.originalDF = None
        self.transformDF = None
        self.inputUrl1 = args["input"][0]["value"]
        self.outputUrl1 = args["output"][0]["value"]

        self.columns = args["param"]["columns"]
        self.k = args["param"]["k"]

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        self.originalDF = self.dataUtil.PyReadHive(self.inputUrl1)

    def execute(self):
        from sklearn.decomposition import PCA
        import pandas as pd
        assert int(self.k) <= self.originalDF.shape[0], "维度需不大于样本数"
        self.logger.info("transforming with %s components" % (str(self.k)))
        pca = PCA(n_components=self.k).fit_transform(self.originalDF[self.columns])
        pca_feature_names = ["pca_" + str(i) for i in range(self.k)]

        self.transformDF = pd.concat([self.originalDF[self.columns], pd.DataFrame(pca, columns=pca_feature_names)],
                                     axis=1)

    def setOut(self):
        self.dataUtil.PyWriteHive(self.transformDF, self.outputUrl1)


class FLDA:
    pass


class SVD:
    pass


class FeatureScale:
    pass


class FeatureSoften:
    pass


class AnomalyDetection:
    pass


class FeatureDiscrete:
    pass


class OneHot:
    pass
