import sys
import inspect
import importlib
import os

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0:2] == "-D":  # Found a "-name value" pair.
            opts[argv[0][1:]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

if __name__ =="__main__":
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
    if cmd_folder not in sys.path:
        sys.path.insert(0, cmd_folder)
    args = getopts(sys.argv)
    print(args)

    # get the second argument from the command line
    # split this into module name and class name
    className = args["DmoduleName"].split(".")[-1]
    moduleName = ".".join(args["DmoduleName"].split(".")[:-1])
    print(moduleName)
    # get pointers to the objects based on the string names
    try:
        type = args["Dtype"]
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
    currentClass = chosenClass(args)
    print(currentClass)
    currentClass.getIn()
    currentClass.execute()
    currentClass.setOut()
