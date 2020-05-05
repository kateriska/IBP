# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: bt.py - arguments parsing and calling of all modules

import sys
import pywt
import LBPGLCMclasif # extraction of vector based on LBP
import artificialNeuralNetwork # classification with ANN
import supportVectorMachines # classification with SVM
import decisionTree # classification with Decision Tree
import SobelLaplacianClasif # extraction of vector based on Sobel and Laplacian operator
import WaveletClasif # extraction of vector based on Wavelet transform
import LBPshow # show histogram and LBP image of input image
import WaveletShow # show Wavelet transform of input image
import SobelLaplacianShow # show results of Sobel and Laplacian operator of input image
import SegmentationShow # show segmentation of input image with use of various tresholding methods

# function for print help for classification commands
def printHelpClasify():
    print("HELP FOR CLASSIFICATION:")
    print("Clasify with LBP: -test lbp -s otsu|gauss|mean -clasif ann|svm|dts -img b|g|r|all")
    print("Clasify with SobelLaplacian: -test sobel -clasif ann|svm|dts -img b|g|r|all")
    print("Clasify with Wavelet: -test wavelet -t <wavelet type> -s otsu|gauss|mean -clasif ann|svm|dts -img b|g|r|all")
    print("Show help for clasify commands: -test help")
    print("Show general help: -help")
    print("Note - Segmentation for SobelLaplacian is implicitly using Otsu thresholding")
    print()
    return

# function for print help for show methods commands
def printHelpShow():
    print("HELP FOR SHOWING PROCESSED SAMPLES:")
    print("Show LBP img: -show lbp [-s otsu|gauss|mean] -img <path>")
    print("Show SobelLaplacian img: -show sobel [-s otsu|gauss|mean] -img <path>")
    print("Show Wavelet: -show wavelet -t <wavelet type> [-s otsu|gauss|mean] -img <path>")
    print("Show Segmented img: -show seg -s otsu|gauss|mean -img <path>")
    print("Show help for show commands: -show help")
    print("Show general help: -help")
    print("Note - Segmentation argument -s is mandatory only for Show Segmented img")
    print()
    return

# function for control of correct arguments
def controlCorrectArguments(arguments_count):
    if (arguments_count == 1 and (sys.argv[1] == "-help")):
        printHelpClasify()
        printHelpShow()
        exit(0)

    elif (sys.argv[1] == "-test"):
        if (arguments_count < 2 or arguments_count > 10):
            sys.stderr.write("ERROR - Wrong count of arguments\n")
            exit(1)
        elif (sys.argv[2] == "help"):
            printHelpClasify()
            exit(0)
        elif ((sys.argv[2] != "lbp") and (sys.argv[2] != "sobel") and (sys.argv[2] != "wavelet")):
            sys.stderr.write("ERROR - Use lbp, sobel, wavelet or help as argument for -test\n")
            exit(1)

        elif ((sys.argv[arguments_count] != "b") and (sys.argv[arguments_count] != "g") and (sys.argv[arguments_count] != "r")  and (sys.argv[arguments_count] != "all")):
            sys.stderr.write("ERROR - Use b (all images with blue light), g (all images with green light), r (all images with red light) or all (mix of images with different lights) as argument for -img\n")
            exit(1)

        elif (sys.argv[2] == "wavelet" and ((sys.argv[3] != "-t") or (sys.argv[5] != "-s") or (sys.argv[7] != "-clasif") or (sys.argv[9] != "-img"))):
            sys.stderr.write("ERROR - Use switches -t, -s, -clasif, -img for clasify with wavelet\n")
            exit(1)

        elif (sys.argv[2] == "lbp" and ((sys.argv[3] != "-s") or (sys.argv[5] != "-clasif") or (sys.argv[7] != "-img"))):
            sys.stderr.write("ERROR - Use switches -s, -clasif, -img for clasify with lbp\n")
            exit(1)

        elif (sys.argv[2] == "sobel" and ((sys.argv[3] != "-clasif") or (sys.argv[5] != "-img"))):
            sys.stderr.write("ERROR - Use switches -clasif, -img for clasify with sobel\n")
            exit(1)
        elif ((sys.argv[2] == "wavelet" and arguments_count != 10) or (sys.argv[2] == "lbp" and arguments_count != 8) or (sys.argv[2] == "sobel" and arguments_count != 6)):
            sys.stderr.write("ERROR - Wrong count of arguments\n")
            exit(1)

        elif (sys.argv[arguments_count - 2] != "ann" and sys.argv[arguments_count - 2] != "svm" and sys.argv[arguments_count - 2] != "dts"):
            sys.stderr.write("ERROR - Use paramatres for -clasif ann, svm or dts\n")
            exit(1)

        elif ((sys.argv[2] == "wavelet" or sys.argv[2] == "lbp") and(sys.argv[arguments_count - 4] != "otsu" and sys.argv[arguments_count - 4] != "gauss" and sys.argv[arguments_count - 4] != "mean")):
            sys.stderr.write("ERROR - Use paramatres for -s otsu, gauss or mean\n")
            exit(1)

    elif (sys.argv[1] == "-show"):
        if (arguments_count < 2 or arguments_count > 8):
            sys.stderr.write("ERROR - Wrong count of arguments\n")
            exit(1)
        elif (sys.argv[2] == "help" and arguments_count == 2):
            printHelpShow()
            exit(0)
        elif ((sys.argv[2] != "lbp") and (sys.argv[2] != "sobel") and (sys.argv[2] != "wavelet") and (sys.argv[2] != "seg")):
            sys.stderr.write("ERROR - Use lbp, sobel, wavelet or help as argument for -show\n")
            exit(1)

        elif(sys.argv[arguments_count - 1] != "-img"):
            sys.stderr.write("ERROR - Use -img for img path in -show\n")
            exit(1)

        elif (sys.argv[2] == "wavelet" and (sys.argv[3] != "-t")):
            sys.stderr.write("ERROR - Use compulsory -t for type of wavelet for show wavelet img\n")
            exit(1)

        elif ("-s" in sys.argv and (("otsu" not in sys.argv) and ("gauss" not in sys.argv) and ("mean" not in sys.argv))):
            sys.stderr.write("ERROR - Use otsu, gauss or mean with -s (segmentation)\n")
            exit(1)
    else:
        sys.stderr.write("ERROR - Use -show, -help or -test, for more information see -help.\n")
        exit(1)


arguments_count = len(sys.argv) - 1
controlCorrectArguments(arguments_count)

# -show commands
if (sys.argv[1] == "-show"):
    input_img = sys.argv[arguments_count]

    # SHOW LBP
    if (sys.argv[2] == "lbp"):
        if (sys.argv[3] == "-s"):
            if (sys.argv[4] == "otsu" or sys.argv[4] == "gauss" or sys.argv[4] == "mean" ):
                segmentation_type = sys.argv[4]
        else:
            segmentation_type = "none"

        LBPshow.showLBP(segmentation_type, input_img)

    # SHOW WAVELET
    elif (sys.argv[2] == "wavelet"):
        if (sys.argv[3] == "-t"):
            wavelet_type = sys.argv[4]
            if (wavelet_type not in pywt.wavelist(kind='discrete')):
                sys.stderr.write("ERROR - Unknown type of wavelet\n")
                exit(1)
            if (sys.argv[5] == "-s"):
                if (sys.argv[6] == "otsu" or sys.argv[6] == "gauss" or sys.argv[6] == "mean" ):
                    segmentation_type = sys.argv[6]
            else:
                segmentation_type = "none"
            WaveletShow.showWavelet(segmentation_type, input_img, wavelet_type)

    # SHOW SOBEL LAPLACIAN
    elif (sys.argv[2] == "sobel"  ):
        if (sys.argv[3] == "-s"):
            if (sys.argv[4] == "otsu" or sys.argv[4] == "gauss" or sys.argv[4] == "mean" ):
                segmentation_type = sys.argv[4]
        else:
            segmentation_type = "none"
        SobelLaplacianShow.showSobelLaplacian(segmentation_type, input_img)

    # SHOW SEGMENTATION
    elif (sys.argv[2] == "seg"  ):
        if (sys.argv[3] == "-s"):
            if (sys.argv[4] == "otsu" or sys.argv[4] == "gauss" or sys.argv[4] == "mean" ):
                segmentation_type = sys.argv[4]
                SegmentationShow.showSegmentation(segmentation_type, input_img)

# -test commands
elif (sys.argv[1] == "-test"):
    if ("-img" in sys.argv and "-clasif" in sys.argv):
        color_type = sys.argv[arguments_count]
        clasif_type = sys.argv[arguments_count - 2]

        # CLASIFY LBP
        if (sys.argv[2] == "lbp" ):
            if (sys.argv[3] == "-s"):
                segmentation_type = sys.argv[4]
                LBPGLCMclasif.vectorLBP(segmentation_type, color_type)

                if (clasif_type == "ann"):
                    artificialNeuralNetwork.clasifyANN("lbp")
                elif (clasif_type == "svm"):
                    supportVectorMachines.clasifySVM("lbp")
                elif (clasif_type == "dts"):
                    decisionTree.clasifyDTS("lbp")

        # CLASIFY SOBEL LAPLACIAN
        elif (sys.argv[2] == "sobel" ):
            SobelLaplacianClasif.vectorSobelLaplacian(color_type)

            if (clasif_type == "ann"):
                artificialNeuralNetwork.clasifyANN("sobel")
            elif (clasif_type == "svm"):
                supportVectorMachines.clasifySVM("sobel")
            elif (clasif_type == "dts"):
                decisionTree.clasifyDTS("sobel")

        # CLASIFY WAVELET
        elif (sys.argv[2] == "wavelet" ):
            if (sys.argv[5] == "-s"):
                segmentation_type = sys.argv[6]
                wavelet_type = sys.argv[4]

                if (wavelet_type not in pywt.wavelist(kind='discrete')):
                    sys.stderr.write("ERROR - Unknown type of wavelet\n")
                    exit(1)

                WaveletClasif.vectorWavelet(segmentation_type, color_type, wavelet_type)

                if (clasif_type == "ann"):
                    artificialNeuralNetwork.clasifyANN("wavelet")
                elif (clasif_type == "svm"):
                    supportVectorMachines.clasifySVM("wavelet")
                elif (clasif_type == "dts"):
                    decisionTree.clasifyDTS("wavelet")
