# Author: Katerina Fortova
# Bachelor's Thesis: Liveness Detection on Touchless Fingerprint Scanner
# Academic Year: 2019/20
# File: artificialNeuralNetwork.py - implementation of Artificial Neural Network for classification of fingerprint liveness

import numpy as np
from matplotlib import pyplot as plt
import math

# class for Artificial Neural Network
class ArtificialNeuralNetwork:
    def __init__(self, trained_vectors, trained_results, vector_weights):
        self.trained_vectors = trained_vectors # extracted vectors of trained images
        self.trained_results = trained_results # known results of trained images (live - 1, fake - 0 )
        self.vector_weights = vector_weights # weights of values from vector, in the beginning randomly initialized
        self.epochs_list = [] # list for count of epochs
        self.error_list = [] # list for saving errors

    def sigmoidFunction(self, x, derivative_value):
        if (derivative_value == False): # sigmoid function
            function_value = (1 / (1 + np.exp(-x)))
        elif (derivative_value == True): # derivative of sigmoid function
            function_value = ((np.exp(-x)) * ((1 + (np.exp(-x)))**(-2)))
        return function_value

    def processData(self):
        vector_weights_product = np.dot(self.trained_vectors, self.vector_weights) # processing of data, multiply vectors with weights
        self.hidden_layer = self.sigmoidFunction(vector_weights_product, False)

    def processBackpropagation(self):
        # compute of delta rule for backpropagation
        self.error_value = self.trained_results - self.hidden_layer
        delta_value = self.error_value * self.sigmoidFunction(self.hidden_layer, True)
        trained_vectors_transpose = self.trained_vectors.T
        vector_delta_product = np.dot(trained_vectors_transpose, delta_value)
        self.vector_weights = self.vector_weights + vector_delta_product # compute of new weights

    def processTraining(self, epochs_count):
        epoch = 0
        while (epoch < epochs_count): # run for count of epochs
            self.processData() # data are processed by ANN
            self.processBackpropagation() # backpropagation of processed data, updating weights
            # get error of epoch and append it to list
            error_absolute = np.absolute(self.error_value)
            average_error = np.average(error_absolute)
            self.error_list.append(average_error)
            self.epochs_list.append(epoch) # add this epoch to list of all of them
            epoch = epoch + 1

    def processTesting(self, tested_vector):
        tested_vector_weight_product = np.dot(tested_vector, self.vector_weights) # predict results of tested vectors in trained ANN
        tested_result = self.sigmoidFunction(tested_vector_weight_product, False)
        return tested_result

def clasifyANN(method_type):
    # csv files with data about vectors for LBP, Sobel and Wavelet methods
    if (method_type == "lbp"):
        trained_vectors = np.genfromtxt('./csvFiles/LBPGLCMtrained.csv',delimiter=",")
        trained_results = np.genfromtxt('./csvFiles/LBPGLCMtrainedResult.csv',dtype=int)
        tested_vectors = np.genfromtxt('./csvFiles/LBPGLCMtested.csv',delimiter=",", usecols=(1,2,3,4,5,6,7,8))
        tested_files = np.genfromtxt('./csvFiles/LBPGLCMtested.csv',delimiter=",", usecols=(0), dtype=None, encoding=None)

    elif (method_type == "sobel"):
        trained_vectors = np.genfromtxt('./csvFiles/SLtrained.csv',delimiter=",")
        trained_results = np.genfromtxt('./csvFiles/SLtrainedResult.csv',dtype=int)
        tested_vectors = np.genfromtxt('./csvFiles/SLtested.csv',delimiter=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
        tested_files = np.genfromtxt('./csvFiles/SLtested.csv',delimiter=",", usecols=(0), dtype=None, encoding=None)

    elif (method_type == "wavelet"):
        trained_vectors = np.genfromtxt('./csvFiles/WaveletTrained.csv',delimiter=",")
        trained_results = np.genfromtxt('./csvFiles/WaveletTrainedResult.csv',dtype=int)
        tested_vectors = np.genfromtxt('./csvFiles/WaveletTested.csv',delimiter=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
        tested_files = np.genfromtxt('./csvFiles/WaveletTested.csv',delimiter=",", usecols=(0), dtype=None, encoding=None)

    trained_vectors_size = trained_vectors.size
    trained_results_size = trained_results.size
    vector_length = int (trained_vectors_size / trained_results_size) # compute length of vector because of length of vector of weights
    print(vector_length)

    mu, sigma = 0, 1 # mean and standard deviation, randomly init weights of ANN
    vector_weights = np.random.normal(mu, sigma, size=(vector_length, 1))
    print(vector_weights)
    trained_results = trained_results[:,np.newaxis]


    artificial_neural_network = ArtificialNeuralNetwork(trained_vectors, trained_results, vector_weights)
    artificial_neural_network.processTraining(500000) # process training for 1 000 000 epochs


    rows = tested_vectors.shape[0]
    cols = tested_vectors.shape[1]
    live_sample = False
    tested_files_index = 0
    correct_clasify = 0
    wrong_clasify = 0
    far_value = 0 # the percentage of identification instances in which unauthorised persons are incorrectly accepted
    frr_value = 0 # the percentage of identification instances in which authorised persons are incorrectly rejected
    live_count = 0 # count of live fingerprints
    fake_count = 0 # count of fake fingerprints
    for cols in tested_vectors:
        live_percent = 0
        fake_percent = 0

        array_testing = cols[:,np.newaxis]

        array_transpose = array_testing.T

        final_prediction = artificial_neural_network.processTesting(array_transpose) # process testing of each vector with unknown result
        final_prediction_str = str(final_prediction)
        final_prediction_str = final_prediction_str[2:-2]

        file_tested_str = str(tested_files.item(tested_files_index))

        final_prediction_float = float(final_prediction_str)
        live_percent = final_prediction_float * 100 # value of liveness in fingeprint
        fake_percent = (1 - final_prediction_float) * 100 # value of fakeness in fingerprint
        if (final_prediction_float >= 0.5 and final_prediction_float <= 1): # decide, whether live or fake have bigger value
            live_sample = True
        else:
            live_sample = False
        if (live_sample == True):
            print("File " + file_tested_str + " is classified as LIVE (LIVE: " + str(round(live_percent, 6)) + " % , FAKE: " + str(round(fake_percent, 6)) + " %)")
        else:
            print("File " + file_tested_str + " is classified as FAKE (LIVE: " + str(round(live_percent, 6)) + " % , FAKE: " + str(round(fake_percent, 6)) + " %)")

        # decide whether the classification is correct or not, based on comparing with name of file
        if (live_sample == True and "live" in file_tested_str) or (live_sample == False and "fake" in file_tested_str):
            correct_clasify += 1
        else:
            wrong_clasify += 1

        # count sum of all tested live and fake fingerprints
        if ("live" in file_tested_str):
            live_count += 1
        else:
            fake_count += 1

        # compute FAR and FRR
        if (live_sample == True and "fake" in file_tested_str):
            far_value += 1
        elif (live_sample == False and "live" in file_tested_str):
            frr_value += 1

        tested_files_index = tested_files_index + 1

    print()
    print("Number of tested live fingerprints: " + str(live_count))
    print("Number of tested fake fingerprints: " + str(fake_count))
    print("Number of correct classifications: " + str(correct_clasify))
    print("Number of wrong classifications: " + str(wrong_clasify))
    accuracy = (100 * correct_clasify) / (correct_clasify + wrong_clasify)
    print("Accuracy: " + str(accuracy) + " %" )
    far_count = (far_value * 100) / fake_count
    frr_count = (frr_value * 100) / live_count
    print("FAR (unauthorised persons are incorrectly accepted): " + str(far_count) + " %")
    print("FRR (authorised persons are incorrectly rejected): " + str(frr_count) + " %")

    # show graph of error of ANN
    plt.figure(figsize=(15,5))
    plt.plot(artificial_neural_network.epochs_list, artificial_neural_network.error_list)
    plt.xlabel('Count of epochs')
    plt.ylabel('Error value')
    plt.show()

    return
