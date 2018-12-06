import sys
import inspect
import importlib
import os
import json
import base64
import logging
import logging.config
import utils
logging.config.fileConfig('logging.ini')


if __name__ == "__main__":
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
    if cmd_folder not in sys.path:
        sys.path.insert(0, cmd_folder)

    def _parse_args(b64str):
        pad = len(b64str) % 4
        b64str += "=" * pad
        return base64.b64decode(b64str)

    args = json.loads(sys.argv[1])
    args2 = json.loads(sys.argv[2])
    #args = json.loads(_parse_args(sys.argv[1]))
    #args2 = json.loads(_parse_args(sys.argv[2]))
    # get the second argument from the command line
    # split this into module name and class name
    className = args["class"].split(".")[-1]
    moduleName = ".".join(args["class"].split(".")[:-1])
    # get pointers to the objects based on the string names
    classType = args["classType"]
    # get ML model type
    modelPath = None
    if len(args["input"]) != 0:
        modelPath = args["input"][0]["value"] if args["input"][0]["type"] == "model" else None
    chosenClass = None

    if classType == "Py.Distributed":
        chosenClass = getattr(importlib.import_module("idsw-spark." + moduleName), className)
    elif classType == "Py.Standalone":
        chosenClass = getattr(importlib.import_module("idsw."+moduleName), className)
    elif classType.startswith("Py"):
        # 对于不指定单机或分布式的机器学习组件，若输入参数含有机器学习模型，判断为单机模型还是分布式模型，分别处理
        if modelPath is not None:
            from sklearn.externals import joblib
            try:
                hdfs = utils.dataUtil(args2)._get_HDFS_connection()
                with hdfs.open(modelPath) as reader:
                    joblib.load(reader)
                    chosenClass = getattr(importlib.import_module("idsw." + moduleName), className)
            except OSError as e:
                if str(e) == "":
                    chosenClass = getattr(importlib.import_module("idsw-spark." + moduleName), className)
        # 如果没有涉及机器学习模型，则认为默认使用单机模块
        else:
            chosenClass = getattr(importlib.import_module("idsw." + moduleName), className)

    # initialize processing class
    currentClass = chosenClass(args, args2)

    logger = logging.getLogger(currentClass.__class__.__name__)
    # get input
    currentClass.getIn()
    logger.info("%s initialization succeeded!" % (moduleName + "." + className))
    # execute
    currentClass.execute()
    logger.info("%s execution succeeded!" % (moduleName + "." + className))
    # write output
    currentClass.setOut()
    logger.info("%s all succeeded!" % (moduleName + "." + className))
