import sys
import inspect
import importlib
import os
import json
import base64
import logging


if __name__ == "__main__":
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
    if cmd_folder not in sys.path:
        sys.path.insert(0, cmd_folder)

    def _parse_args(str):
        pad = len(str) % 4
        str += "=" * pad
        return base64.b64decode(str)

    args = json.loads(_parse_args(sys.argv[1]))
    args2 = json.loads(_parse_args(sys.argv[2]))
    # get the second argument from the command line
    # split this into module name and class name
    className = args["class"].split(".")[-1]
    moduleName = ".".join(args["class"].split(".")[:-1])
    # get pointers to the objects based on the string names
    type = args["classType"]

    if type == "Py.Distributed":
        chosenClass = getattr(importlib.import_module("idsw." + moduleName), "Spark" + className)
    elif type == "Py.Standalone":
        chosenClass = getattr(importlib.import_module("idsw."+moduleName), "Py"+className)
    elif type.startswith("Py"):
        chosenClass = getattr(importlib.import_module("idsw." + moduleName), "Py" + className)
    # initialize processing class
    currentClass = chosenClass(args, args2)
    logger = logging.getLogger(currentClass.__class__.__name__)
    currentClass.getIn()
    logger.info("%s initialization succeeded!" %(moduleName + "." + className))
    currentClass.execute()
    logger.info("%s execution succeeded!" % (moduleName + "." + className))
    currentClass.setOut()
    logger.info("%s all succeeded!" % (moduleName + "." + className))
