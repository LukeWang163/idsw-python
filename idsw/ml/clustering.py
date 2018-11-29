#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.ml.clustering.py
# @Desc    : Scripts for initializing clustering models. 机器学习->模型初始化->聚类
import logging
import logging.config
import utils

logging.config.fileConfig('logging.ini')


class KMeans:
    def __init__(self, args, args2):
        """
        Standalone version for initializing KMeans clustering
        @param args: dict
        K: int
        init: string one of "k-means++" and "random"
        n_init: int
        max_iter: int
        tol: float
        """
        # init logging
        self.logger = logging.getLogger(self.__class__.__name__)

        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]
        self.model = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        return

    def execute(self):
        self.logger.info("using scikit-learn for K-means")
        # 聚类中心个数
        k = int(self.param["K"])
        # 初始化方法
        init = self.param["init"]
        if init == "k-means":
            init = "k-means++"
        # 运行次数
        n_init = int(self.param["n_init"])
        # 单次训练最大迭代次数
        max_iter = int(self.param["max_iter"])
        # 容忍度
        tol = int(self.param["tol"])

        # 初始化模型
        from sklearn.cluster import KMeans
        self.model = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, tol=tol)

    def setOut(self):
        self.logger.info("saving initialized standalone KMeans model to %s" % self.outputUrl1)
        self.dataUtil.PyWriteModel(self.model, self.outputUrl1)


class DBSCAN:
    def __init__(self, args, args2):
        """
        Standalone version for initializing KMeans clustering
        @param args: dict
        K: int
        init: string one of "k-means++" and "random"
        n_init: int
        max_iter: int
        tol: float
        """
        # init logging
        self.logger = logging.getLogger(self.__class__.__name__)

        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]
        self.model = None

        self.dataUtil = utils.dataUtil(args2)

    def getIn(self):
        return

    def execute(self):
        self.logger.debug("using scikit-learn for K-means")
        # 聚类中心个数
        k = int(self.param["K"])
        # 初始化方法
        init = self.param["init"]
        # 运行次数
        n_init = int(self.param["n_init"])
        # 单次训练最大迭代次数
        max_iter = int(self.param["max_iter"])
        # 容忍度
        tol = int(self.param["tol"])

        # 初始化模型
        from sklearn.cluster import KMeans
        self.model = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, tol=tol)

    def setOut(self):
        self.logger.info("saving initialized distributed KMeans model to %s" % self.outputUrl1)
        self.dataUtil.PyWriteModel(self.model, self.outputUrl1)


class Kohogen:
    pass
