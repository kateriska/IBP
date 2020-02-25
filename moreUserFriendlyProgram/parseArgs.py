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
import SobelLaplacianShow
import SegmentationShow


def printHelpClasify():
    print("HELP:")
    print("Clasify with LBP: -test lbp -s otsu|gauss|mean -clasif ann|svm|clf|myclf -img b|g|r|all")
    print("Clasify with SobelLaplacian: -test sobel -clasif ann|svm|clf|myclf -img b|g|r|all")
    print("Clasify with Wavelet: -test wavelet -t <type of wavelet, eg. db2> -s otsu|gauss|mean -clasif ann|svm|clf|myclf -img b|g|r|all")
    print("Note - Only segmentation for Sobel Clasify is Otsu, because of better results")
    return

def printHelpShow():
    print("HELP:")
    print("Show LBP img: -show lbp [-s otsu|gauss|mean] -img path")
    print("Show SobelLaplacian img: -show sobel [-s otsu|gauss|mean] -img path")
    print("Show Wavelet: -show wavelet -t <type of wavelet, eg. db2> [-s otsu|gauss|mean] -img path")
    print("Show Segmented img: -show seg -s otsu|gauss|mean -img path")
    print("Note - Segmentation argument -s is mandatory")
    return


arguments_count = len(sys.argv) - 1
if (sys.argv[1] == "-test"):
    if (sys.argv[2] == "help"):
        printHelpClasify()
        exit(0)
    elif ((sys.argv[2] != "lbp") and (sys.argv[2] != "sobel") and (sys.argv[2] != "wavelet")):
        sys.stderr.write("ERROR - Use lbp, sobel, wavelet or help as argument for -test\n")
        #printHelpClasify()
        exit(1)

    elif ((sys.argv[arguments_count] != "b") and (sys.argv[arguments_count] != "g") and (sys.argv[arguments_count] != "r")  and (sys.argv[arguments_count] != "all")):
        sys.stderr.write("ERROR - Use b (all images with blue light), g (all images with green light), r (all images with red light) or all (mix of images with different lights) as argument for -img\n")
        #printHelpClasify()
        exit(1)

    elif (sys.argv[2] == "wavelet" and ((sys.argv[3] != "-t") or (sys.argv[5] != "-s") or (sys.argv[7] != "-clasif") or (sys.argv[9] != "-img"))):
        sys.stderr.write("ERROR - Use switches -t, -s, -clasif, -img for clasify with wavelet\n")
        #printHelpClasify()
        exit(1)

    elif (sys.argv[2] == "lbp" and ((sys.argv[3] != "-s") or (sys.argv[5] != "-clasif") or (sys.argv[7] != "-img"))):
        sys.stderr.write("ERROR - Use switches -s, -clasif, -img for clasify with lbp\n")
        #printHelpClasify()
        exit(1)

    elif (sys.argv[2] == "sobel" and ((sys.argv[3] != "-clasif") or (sys.argv[5] != "-img"))):
        sys.stderr.write("ERROR - Use switches -clasif, -img for clasify with sobel\n")
        #printHelpClasify()
        exit(1)
    elif (sys.argv[2] == "wavelet" and arguments_count != 10):
        sys.stderr.write("ERROR - Wrong count of arguments\n")
        exit(1)
    elif (sys.argv[2] == "lbp" and arguments_count != 8):
        sys.stderr.write("ERROR - Wrong count of arguments\n")
        exit(1)
    elif (sys.argv[2] == "sobel" and arguments_count != 6):
        sys.stderr.write("ERROR - Wrong count of arguments\n")
        exit(1)

    elif (sys.argv[arguments_count - 2] != "ann" and sys.argv[arguments_count - 2] != "svm" and sys.argv[arguments_count - 2] != "clf" and sys.argv[arguments_count - 2] != "myclf"):
        sys.stderr.write("ERROR - Use paramatres for -clasif ann, svm, clf or myclf\n")
        #printHelpClasify()
        exit(1)

    elif ((sys.argv[2] == "wavelet" or sys.argv[2] == "lbp") and(sys.argv[arguments_count - 4] != "otsu" and sys.argv[arguments_count - 4] != "gauss" and sys.argv[arguments_count - 4] != "mean")):
        sys.stderr.write("ERROR - Use paramatres for -s otsu, gauss or mean\n")
        #printHelpClasify()
        exit(1)

if (sys.argv[1] == "-show"):
    if (sys.argv[2] == "help"):
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
    elif (sys.argv[2] == "wavelet"):
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

    # SHOW SOBEL LAPLACIAN
    elif (sys.argv[2] == "sobel"  ):
        if (sys.argv[3] == "-s"):
            if (sys.argv[4] == "otsu" or sys.argv[4] == "gauss" or sys.argv[4] == "mean" ):
                segmentation_type = sys.argv[4]
                print(segmentation_type)
        else:
            segmentation_type = "none"
            print(segmentation_type)
        SobelLaplacianShow.showSobelLaplacian(segmentation_type, input_img)

    elif (sys.argv[2] == "seg"  ):
        if (sys.argv[3] == "-s"):
            if (sys.argv[4] == "otsu" or sys.argv[4] == "gauss" or sys.argv[4] == "mean" ):
                segmentation_type = sys.argv[4]
                print(segmentation_type)

                SegmentationShow.showSegmentation(segmentation_type, input_img)

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
