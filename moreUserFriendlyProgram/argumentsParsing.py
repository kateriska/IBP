import getopt
import sys
import LBPGLCMexperiment
import NeuralNetworkClasify
import SVMexperiment
import decisionTree
import NewClassificationTree
import SobelLaplacianExperiment
import WaveletExperiment


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

        # LBP
        if (sys.argv[2] == "lbp" ):
            if (sys.argv[3] == "-s"):
                if (sys.argv[4] == "otsu" or sys.argv[4] == "gauss" or sys.argv[4] == "mean" ):
                    segmentation_type = sys.argv[4]
                    print(segmentation_type)
                    print(color_type)
                    LBPGLCMexperiment.vectorLBP(segmentation_type, color_type)

                    if (clasif_type == "ann"):
                        NeuralNetworkClasify.clasifyANN("lbp")

                    elif (clasif_type == "svm"):
                        SVMexperiment.clasifySVM("lbp")

                    elif (clasif_type == "clf"):
                        decisionTree.clasifyCLF("lbp")
                    elif (clasif_type == "myclf"):
                        NewClassificationTree.clasifyMyCLF("lbp")

        # SOBEL
        elif (sys.argv[2] == "sobel" ):
            #if (sys.argv[3] == "-s"):
                #segmentation_type = "otsu"
            print(color_type)
            SobelLaplacianExperiment.vectorSobelLaplacian(color_type)

            if (clasif_type == "ann"):
                NeuralNetworkClasify.clasifyANN("sobel")

            elif (clasif_type == "svm"):
                SVMexperiment.clasifySVM("sobel")

            elif (clasif_type == "clf"):
                decisionTree.clasifyCLF("sobel")

            elif (clasif_type == "myclf"):
                NewClassificationTree.clasifyMyCLF("sobel")




        # WAVELET
        elif (sys.argv[2] == "wavelet" ):
            if (sys.argv[5] == "-s"):
                if ((sys.argv[6] == "otsu" or sys.argv[6] == "gauss" or sys.argv[6] == "mean") and  sys.argv[3] == "-t"):
                    segmentation_type = sys.argv[6]
                    print(segmentation_type)
                    wavelet_type = sys.argv[4]
                    print(wavelet_type)
                    WaveletExperiment.vectorWavelet(segmentation_type, color_type, wavelet_type)

                    if (clasif_type == "ann"):
                        NeuralNetworkClasify.clasifyANN("wavelet")

                    elif (clasif_type == "svm"):
                        SVMexperiment.clasifySVM("wavelet")

                    elif (clasif_type == "clf"):
                        decisionTree.clasifyCLF("wavelet")

                    elif (clasif_type == "myclf"):
                        NewClassificationTree.clasifyMyCLF("wavelet")
            #if (sys.argv[3] == "-t"):
                #wavelet_type = sys.argv[4]
                #print(wavelet_type)
