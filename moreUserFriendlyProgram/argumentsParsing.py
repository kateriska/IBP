import getopt
import sys


print("the script has the name %s" % (sys.argv[0]))
print(sys.argv[1])
arguments_count = len(sys.argv) - 1

if (sys.argv[1] == "-show"):
    print("To do")

elif (sys.argv[1] == "-test"):
    print("Clasify")
    if ("-img" in sys.argv and "-clasif" in sys.argv):
        color_type = sys.argv[arguments_count]
        print(color_type)
        clasif_type = sys.argv[arguments_count - 2]
        print(clasif_type)

        if (sys.argv[2] == "lbp" ):
            if (sys.argv[3] == "-s"):
                if (sys.argv[4] == "otsu" or sys.argv[4] == "gauss" or sys.argv[4] == "mean" ):
                    segmentation_type = sys.argv[4]
                    print(segmentation_type)
