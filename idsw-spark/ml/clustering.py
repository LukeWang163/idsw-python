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

        # init parameters
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]
        self.model = None

        self.logger.info("initializing SparkSession")
        # init SparkSession
        self.spark = utils.init_spark()

    def getIn(self):
        return

    def execute(self):
        from pyspark.ml.clustering import KMeans
        from pyspark.ml import Pipeline

        # 聚类中心个数
        k = int(self.param["K"])
        # 初始化方法
        init = self.param["init"]
        if init == "k-means":
            init = "k-means||"
        # 运行次数
        n_init = int(self.param["n_init"])
        # 单次训练最大迭代次数
        max_iter = int(self.param["max_iter"])
        # 容忍度
        tol = int(self.param["tol"])

        # 以Pipeline的模式初始化模型，方便统一接口加载模型
        self.logger.info("initializing model")
        self.model = Pipeline(stages=[
            KMeans(k=k, initMode=init, initSteps=n_init, maxIter=max_iter, tol=tol)
        ])

    def setOut(self):
        self.logger.info("saving model to %s" % self.outputUrl1)
        self.model.write().overwrite().save(self.outputUrl1)
