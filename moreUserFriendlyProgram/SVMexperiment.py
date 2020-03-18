# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: SVMexperiment.py - clasificcation with Support Vector Machines (SVM)

import numpy as np
import matplotlib.pyplot as plt
from numpy import newaxis
from sklearn import svm

# function for classification with SVM
def clasifySVM(method_type):
    # csv files with data about vectors for LBP, Sobel and Wavelet methods
    if (method_type == "lbp"):
        trained_vectors = np.genfromtxt('LBPGLCMTrained.csv',delimiter=",")
        trained_results = np.genfromtxt('LBPGLCMresultTrained.csv',dtype=int)
        tested_vectors = np.genfromtxt('LBPGLCMtested.csv',delimiter=",", usecols=(1,2,3,4,5,6,7,8))
        tested_files = np.genfromtxt('LBPGLCMtested.csv',delimiter=",", usecols=(0), dtype=None, encoding=None)

    elif (method_type == "sobel"):
        trained_vectors = np.genfromtxt('SLTrained.csv',delimiter=",")
        trained_results = np.genfromtxt('SLTrainedResult.csv',dtype=int)
        tested_vectors = np.genfromtxt('SLTested.csv',delimiter=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
        tested_files = np.genfromtxt('SLTested.csv',delimiter=",", usecols=(0), dtype=None, encoding=None)

    elif (method_type == "wavelet"):
        trained_vectors = np.genfromtxt('WaveletTrained.csv',delimiter=",")
        trained_results = np.genfromtxt('WaveleTrainedResult.csv',dtype=int)
        tested_vectors = np.genfromtxt('WaveletTested.csv',delimiter=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
        tested_files = np.genfromtxt('WaveletTested.csv',delimiter=",", usecols=(0), dtype=None, encoding=None)

    # function for Support Vector Machines from sklearn library
    clf = svm.SVC()
    clf.fit(trained_vectors, trained_results) # train SVM with trained vectors and their results
    prediction = clf.predict(tested_vectors) # predict results of tested vectors
    print(prediction)
    print(tested_files)

    i = 0
    live_sample = False
    correct_clasify = 0
    wrong_clasify = 0
    far_value = 0 # the percentage of identification instances in which unauthorised persons are incorrectly accepted
    frr_value = 0 # the percentage of identification instances in which authorised persons are incorrectly rejected
    for cols in tested_files:
        print(cols)
        print(prediction[i])

        # get final predicted value for fingerprint from vector of predictions
        if (prediction[i] == 1):
            live_sample = True
        elif (prediction[i] == 0):
            live_sample = False

        # compare predicted value with name of file
        if (live_sample == True and "live" in cols) or (live_sample == False and "fake" in cols):
            correct_clasify += 1
        else:
            wrong_clasify += 1

        # compute FAR and FRR
        if (live_sample == True and "fake" in cols):
            far_value += 1
        elif (live_sample == False and "live" in cols):
            frr_value += 1


        i = i + 1

    print("Number of correct classifications: " + str(correct_clasify))
    print("Number of wrong classifications: " + str(wrong_clasify))
    accuracy = (100 * correct_clasify) / (correct_clasify + wrong_clasify)
    print("Accuracy: " + str(accuracy) + " %" )
    far_count = (far_value * 100) / (correct_clasify + wrong_clasify)
    frr_count = (frr_value * 100) / (correct_clasify + wrong_clasify)
    print("FAR (the percentage of identification instances in which unauthorised persons are incorrectly accepted): " + str(far_count) + " %")
    print("FRR (the percentage of identification instances in which authorised persons are incorrectly rejected): " + str(frr_count) + " %")

    return
