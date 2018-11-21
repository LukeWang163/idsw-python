import sys
import inspect
import importlib
import os
import json

if __name__ =="__main__":
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
    if cmd_folder not in sys.path:
        sys.path.insert(0, cmd_folder)
    args = json.loads(sys.argv[1])
    print(args)
    args2 = sys.argv[2]

    # get the second argument from the command line
    # split this into module name and class name
    className = args["class"].split(".")[-1]
    moduleName = ".".join(args["class"].split(".")[:-1])
    print(moduleName)
    # get pointers to the objects based on the string names
    try:
        type = args["classType"]
        print(type)
        if type == "Py.Standalone":
            chosenClass = getattr(importlib.import_module("idsw."+moduleName), "Py"+className)
            print(chosenClass)
        elif type == "Py.Distributed":
            chosenClass = getattr(importlib.import_module("idsw." + moduleName), "Spark" + className)
    except KeyError as e:
        print(e)
        chosenClass = getattr(importlib.import_module("idsw." + moduleName), "Py" + className)

    # initialize processing class
    currentClass = chosenClass(args, args2)
    currentClass.getIn()
    currentClass.execute()
    currentClass.setOut()
    print("Succeeded!")
