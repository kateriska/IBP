import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
from numpy import newaxis
from sklearn import svm

inputs = np.genfromtxt('WaveletTrained.csv',delimiter=",")
outputs = np.genfromtxt('WaveleTrainedResult.csv',dtype=int)
#print(outputs)

clf = svm.SVC()
clf.fit(inputs, outputs)

tested_values = np.genfromtxt('WaveletTested.csv',delimiter=",", usecols=(1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12))
prediction = clf.predict(tested_values)
print(prediction)

tested_files = np.genfromtxt('WaveletTested.csv',delimiter=",", usecols=(0), dtype=None, encoding=None)
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

    if (prediction[i] == 1):
        live_sample = True
    elif (prediction[i] == 0):
        live_sample = False

    if (live_sample == True and "live" in cols) or (live_sample == False and "fake" in cols):
        correct_clasify += 1
    else:
        wrong_clasify += 1

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
