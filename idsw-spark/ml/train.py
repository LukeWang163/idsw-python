#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/19
# @Author  : Luke
# @File    : idsw.ml.train.py
# @Desc    : Scripts for initializing binary classification models. 机器学习->模型训练
import utils
import logging


class TrainModel:
    def __init__(self, args, args2):
        """
        Spark version for training model
        @param args: dict
        featureCols: list
        labelCol: String
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.originalDF = None
        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.param = args["param"]
        self.model = None
        self.pipelineModel = None

        self.logger.info("initializing SparkSession")
        self.spark = utils.init_spark()

    def getIn(self):
        self.logger.debug("using PySpark")
        from pyspark.ml import Pipeline

        self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl2, self.spark)
        self.model = Pipeline.load(self.inputUrl1).getStages()[0]

    def execute(self):
        featureCols = self.param["features"]
        labelCol = self.param["label"]

        # 训练Spark等模型
        from pyspark.ml import Pipeline
        import pyspark.ml.clustering
        from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString
        from pyspark.sql.types import StringType
        print(type(self.model))
        print(self.inputUrl1)
        if not isinstance(self.model, pyspark.ml.clustering.KMeans):

            if "Binary" in self.inputUrl1:
                self.logger.info("training binary classification model")
                if self.originalDF.select(labelCol).distinct().count() != 2:
                    self.logger.error("training data has more than 2 classes. Exiting...")
                    import sys
                    sys.exit(0)
                else:
                    if isinstance(self.originalDF.schema[labelCol].dataType, StringType):
                        # 使用StringIndexer把标签列转为数值类型，使用IndexToString转回
                        self.model.setParams(featuresCol="features", labelCol="indexedLabel")
                        # 使用VectorAssembler将特征列聚合成一个DenseVector
                        pipeline = Pipeline(stages=[VectorAssembler(inputCols=featureCols, outputCol="features"),
                                                    StringIndexer(inputCol=labelCol, outputCol="indexedLabel"),
                                                    self.model,
                                                    IndexToString(inputCol="indexedLabel", outputCol="originalLabel")])
                    else:
                        self.model.setParams(featuresCol="features", labelCol=labelCol)
                        # 使用VectorAssembler将特征列聚合成一个DenseVector
                        pipeline = Pipeline(stages=[VectorAssembler(inputCols=featureCols, outputCol="features"),
                                                    self.model])
                    self.pipelineModel = pipeline.fit(self.originalDF)

            elif "Multi" in self.inputUrl1:
                self.logger.info("training multi-class classification model")
                if isinstance(self.originalDF.schema[labelCol].dataType, StringType):
                    # 使用StringIndexer把标签列转为数值类型，使用IndexToString转回
                    self.model.setParams(featuresCol="features", labelCol="label")
                    # 使用VectorAssembler将特征列聚合成一个DenseVector
                    pipeline = Pipeline(stages=[VectorAssembler(inputCols=featureCols, outputCol="features"),
                                                StringIndexer(inputCol=labelCol, outputCol="label"),
                                                self.model,
                                                IndexToString(inputCol="label", outputCol="originalLabel")])
                else:
                    self.model.setParams(featuresCol="features", labelCol=labelCol)
                    # 使用VectorAssembler将特征列聚合成一个DenseVector
                    pipeline = Pipeline(stages=[VectorAssembler(inputCols=featureCols, outputCol="features"),
                                                self.model])
                self.pipelineModel = pipeline.fit(self.originalDF)

            elif "Reg" in self.inputUrl1:
                self.logger.info("training regression model")
                self.model.setParams(featuresCol="features", labelCol=labelCol)
                # 使用VectorAssembler将特征列聚合成一个DenseVector
                pipeline = Pipeline(stages=[VectorAssembler(inputCols=featureCols, outputCol="features"),
                                            self.model])
                self.pipelineModel = pipeline.fit(self.originalDF)

            else:
                self.logger.error("not supported")
                import sys
                sys.exit(0)

        else:
            self.logger.error("not supported")
            import sys
            sys.exit(0)

    def setOut(self):
        self.logger.info("saving trained distributed model to %s" % self.outputUrl1)
        self.pipelineModel.write().overwrite().save(self.outputUrl1)


class TrainClustering:
    def __init__(self, args, args2):
        """
        Spark version for training clustering model
        @param args: dict
        featureCols: list
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        self.originalDF = None
        self.transformDF = None
        self.inputUrl1 = args["input"][0]["value"]
        self.inputUrl2 = args["input"][1]["value"]
        self.outputUrl1 = args["output"][0]["value"]
        self.outputUrl2 = args["output"][1]["value"]
        self.featureCols = args["param"]["features"]
        self.model = None
        self.pipelineModel = None
        self.logger.debug("initializing SparkSession")
        self.spark = utils.init_spark()

    def getIn(self):
        # 训练Spark等模型
        from pyspark.ml import Pipeline

        self.originalDF = utils.dataUtil.SparkReadHive(self.inputUrl2, self.spark)
        self.model = Pipeline.load(self.inputUrl1).getStages()[0]

    def execute(self):
        # 训练Spark等模型
        from pyspark.ml import Pipeline
        from pyspark.ml.feature import VectorAssembler
        # 使用VectorAssembler将特征列聚合成一个DenseVector
        self.logger.info("training distributed clustering model")
        self.model.setParams(featuresCol="features")
        pipeline = Pipeline(stages=[VectorAssembler(inputCols=self.featureCols, outputCol="features"),
                                    self.model])
        self.pipelineModel = pipeline.fit(self.originalDF)
        self.transformDF = self.pipelineModel.transform(self.originalDF).select(*self.featureCols, "prediction")

    def setOut(self):
        self.logger.info(f"saving trained distributed clustering model to {self.outputUrl1}")
        self.pipelineModel.write().overwrite().save(self.outputUrl1)
        self.logger.info(f"saving fit_transformed data to {self.outputUrl2}")
        utils.dataUtil.SparkWriteHive(self.transformDF, self.outputUrl2)


class TuneHyperparameter:
    pass
