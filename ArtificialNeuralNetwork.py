import numpy as np
from matplotlib import pyplot as plt
import math

trained_vectors = np.genfromtxt('WaveletTrained.csv',delimiter=",")
trained_results = np.genfromtxt('WaveleTrainedResult.csv',dtype=int)
#print(trained_vectors.size)
#print(trained_results.size)
trained_vectors_size = trained_vectors.size
trained_results_size = trained_results.size
vector_length = int (trained_vectors_size / trained_results_size)
print(vector_length)
#vector_weights = np.full((vector_length, 1), 0.1)
#vector_weights = np.random.uniform(low=0.0, high=0.1, size=(vector_length, 1))
mu, sigma = 0, 1 # mean and standard deviation
vector_weights = np.random.normal(mu, sigma, size=(vector_length, 1))
print(vector_weights)
trained_results = trained_results[:,np.newaxis]
#print(vector_weights)

class ArtificialNeuralNetwork:
    def __init__(self, trained_vectors, trained_results, vector_weights):
        self.trained_vectors = trained_vectors
        self.trained_results = trained_results
        self.vector_weights = vector_weights
        self.epochs_list = []
        self.error_list = []

    def sigmoidFunction(self, x, derivative_value):
        if (derivative_value == False):
            function_value = (1 / (1 + np.exp(-x)))
        elif (derivative_value == True):
            function_value = ((np.exp(-x)) * ((1 + (np.exp(-x)))**(-2)))
        return function_value

    def processData(self):
        vector_weights_product = np.dot(self.trained_vectors, self.vector_weights)
        self.hidden_layer = self.sigmoidFunction(vector_weights_product, False)

    def processBackpropagation(self):
        self.error_value = self.trained_results - self.hidden_layer
        delta_value = self.error_value * self.sigmoidFunction(self.hidden_layer, True)
        trained_vectors_transpose = self.trained_vectors.T
        vector_delta_product = np.dot(trained_vectors_transpose, delta_value)
        self.vector_weights = self.vector_weights + vector_delta_product

    def processTraining(self, epochs_count):
        epoch = 0
        while (epoch < epochs_count):
            self.processData()
            self.processBackpropagation()
            error_absolute = np.absolute(self.error_value)
            average_error = np.average(error_absolute)
            self.error_list.append(average_error)
            self.epochs_list.append(epoch)
            epoch = epoch + 1

    def processTesting(self, tested_vector):
        tested_vector_weight_product = np.dot(tested_vector, self.vector_weights)
        tested_result = self.sigmoidFunction(tested_vector_weight_product, False)
        return tested_result

artificial_neural_network = ArtificialNeuralNetwork(trained_vectors, trained_results, vector_weights)
artificial_neural_network.processTraining(1000000)

tested_vectors = np.genfromtxt('WaveletTested.csv',delimiter=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
print(tested_vectors)
tested_files = np.genfromtxt('WaveletTested.csv',delimiter=",", usecols=(0), dtype=None, encoding=None)


rows = tested_vectors.shape[0]
cols = tested_vectors.shape[1]
#all_results_arr = np.zeros((40, 2))
live_sample = False
tested_files_index = 0
correct_clasify = 0
wrong_clasify = 0
far_value = 0 # the percentage of identification instances in which unauthorised persons are incorrectly accepted
frr_value = 0 # the percentage of identification instances in which authorised persons are incorrectly rejected
for cols in tested_vectors:
        live_percent = 0
        fake_percent = 0

        array_testing = cols[:,np.newaxis]

        array_transpose = array_testing.T

        final_prediction = artificial_neural_network.processTesting(array_transpose)
        final_prediction_str = str(final_prediction)
        final_prediction_str = final_prediction_str[2:-2]

        file_tested_str = str(tested_files.item(tested_files_index))

        final_prediction_float = float(final_prediction_str)
        live_percent = final_prediction_float * 100
        fake_percent = (1 - final_prediction_float) * 100
        if (final_prediction_float >= 0.5 and final_prediction_float <= 1):
            live_sample = True
        else:
            live_sample = False
        if (live_sample == True):
            print("File " + file_tested_str + " is classified as LIVE (LIVE: " + str(round(live_percent, 6)) + " % , FAKE: " + str(round(fake_percent, 6)) + " %)")
        else:
            print("File " + file_tested_str + " is classified as FAKE (LIVE: " + str(round(live_percent, 6)) + " % , FAKE: " + str(round(fake_percent, 6)) + " %)")
        #all_results_arr = np.append(all_results_arr, np.array([[file_tested_str, final_prediction_str]]), axis=0)

        if (live_sample == True and "Image" in file_tested_str) or (live_sample == False and "fake" in file_tested_str):
            correct_clasify += 1
        else:
            wrong_clasify += 1

        if (live_sample == True and "fake" in file_tested_str):
            far_value += 1
        elif (live_sample == False and "Images" in file_tested_str):
            frr_value += 1

        tested_files_index = tested_files_index + 1

print("Number of correct classifications: " + str(correct_clasify))
print("Number of wrong classifications: " + str(wrong_clasify))
accuracy = (100 * correct_clasify) / (correct_clasify + wrong_clasify)
print("Accuracy: " + str(accuracy) + " %" )
far_count = (far_value * 100) / (correct_clasify + wrong_clasify)
frr_count = (frr_value * 100) / (correct_clasify + wrong_clasify)
print("FAR (the percentage of identification instances in which unauthorised persons are incorrectly accepted): " + str(far_count) + " %")
print("FRR (the percentage of identification instances in which authorised persons are incorrectly rejected): " + str(frr_count) + " %")

plt.figure(figsize=(15,5))
plt.plot(artificial_neural_network.epochs_list, artificial_neural_network.error_list)
plt.xlabel('Epochs')
plt.ylabel('Error value')
plt.show()
