import getopt
import sys
import LBPGLCMexperiment
import NeuralNetworkClasify
import SVMexperiment
import decisionTree
import NewClassificationTree
import SobelLaplacianExperiment
import WaveletExperiment
import LBPshow
import showWTransform


print("the script has the name %s" % (sys.argv[0]))
print(sys.argv[1])
arguments_count = len(sys.argv) - 1

if (sys.argv[1] == "-test"):
    if ((sys.argv[2] != "lbp") and (sys.argv[2] != "sobel") and (sys.argv[2] != "wavelet")):
        sys.stderr.write("ERROR - Use lbp, sobel or wavelet as argument for -test\n")
        sys.stderr.write("HELP:\n")
        sys.stderr.write("Clasify with LBP: -test lbp -s otsu|gauss|mean -clasif ann|svm|clf|myclf -img b|g|r|all\n")
        sys.stderr.write("Clasify with SobelLaplacian: -test sobel -clasif ann|svm|clf|myclf -img b|g|r|all\n")
        sys.stderr.write("Clasify with Wavelet: -test wavelet -t <type of wavelet, eg. db2> -s otsu|gauss|mean -clasif ann|svm|clf|myclf -img b|g|r|all\n")
        exit(1)

    if ((sys.argv[arguments_count] != "b") or (sys.argv[arguments_count] != "g") or (sys.argv[arguments_count] != "r")  or (sys.argv[arguments_count] != "all")):
        sys.stderr.write("ERROR - Use b (all images with blue light), g (all images with green light), r (all images with red light) or all (mix of images with different lights) as argument for -img\n")
        sys.stderr.write("HELP:\n")
        sys.stderr.write("Clasify with LBP: -test lbp -s otsu|gauss|mean -clasif ann|svm|clf|myclf -img b|g|r|all\n")
        sys.stderr.write("Clasify with SobelLaplacian: -test sobel -clasif ann|svm|clf|myclf -img b|g|r|all\n")
        sys.stderr.write("Clasify with Wavelet: -test wavelet -t <type of wavelet, eg. db2> -s otsu|gauss|mean -clasif ann|svm|clf|myclf -img b|g|r|all\n")
        exit(1)

    #if (sys.argv[2] == "wavelet" and ((sys.argv[3] != "-t") or (sys.argv[5] != "-s") or (sys.argv[7] == "-clasif") or (sys.argv[9] == "-img"))):

    #if (sys.argv[2] == "lbp" and ((sys.argv[3] != "-s") or (sys.argv[5] != "-clasif") or (sys.argv[7] == "-img"))):


if (sys.argv[1] == "-show"):
    print("To do")
    input_img = sys.argv[arguments_count]
    print(input_img)
    # SHOW LBP
    if (sys.argv[2] == "lbp"):
        if (sys.argv[3] == "-s"):
            if (sys.argv[4] == "otsu" or sys.argv[4] == "gauss" or sys.argv[4] == "mean" ):
                segmentation_type = sys.argv[4]
                print(segmentation_type)
            else:
                segmentation_type = "none"

            LBPshow.showLBP(segmentation_type, input_img)

    # SHOW WAVELET
    elif (sys.argv[2] == "wavelet"  ):
        if (sys.argv[3] == "-t"):
            wavelet_type = sys.argv[4]
            print(wavelet_type)
            if (sys.argv[5] == "-s"):
                if (sys.argv[6] == "otsu" or sys.argv[6] == "gauss" or sys.argv[6] == "mean" ):
                    segmentation_type = sys.argv[6]
                    print(segmentation_type)
                else:
                    segmentation_type = "none"
                showWTransform.showWavelet(segmentation_type, input_img, wavelet_type)


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
