import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
from numpy import newaxis

# input data
'''
inputs = np.array([[0.27747, 0.03639, 0.03105, 0.95109],
[0.298, 0.03018, 0.02454, 0.94328],
[0.18637, 0.05777, 0.05405, 0.99781],
[0.32926, 0.03703, 0.03114, 0.89857],
[0.21429, 0.05776, 0.04625, 0.9777],
[0.25761, 0.04447, 0.0362, 0.95772],
[0.23778, 0.03971, 0.03451, 0.984],
[0.28745, 0.08865, 0.08059, 0.83931],
[0.25852, 0.05241, 0.04598, 0.93909],
[0.25376, 0.0857, 0.06896, 0.88758],
[0.27915, 0.0486, 0.04188, 0.92637],
[0.31839, 0.05152, 0.04598, 0.88011],
[0.26537, 0.04928, 0.0425, 0.93885],
[0.19722, 0.08137, 0.05926, 0.95815],
[0.20316, 0.06479, 0.05801, 0.97004],
[0.26682, 0.05204, 0.04515, 0.93199],
[0.29266, 0.05254, 0.04663, 0.90417],
[0.22077, 0.05382, 0.05073, 0.97068],
[0.27292, 0.06918, 0.06056, 0.89334],
[0.21396, 0.04814, 0.048, 0.9859],
[0.26803, 0.07278, 0.05676, 0.89843],
[0.21924, 0.05476, 0.03863, 0.98337],
[0.34269, 0.03285, 0.02612, 0.89434],
[0.24913, 0.03265, 0.02866, 0.98556],
[0.20126, 0.05606, 0.04037, 0.99831],
[0.23798, 0.08937, 0.06932, 0.89933],
[0.1673, 0.09378, 0.05497, 0.97995],
[0.23104, 0.07931, 0.07685, 0.9088],
[0.25652, 0.06296, 0.06117, 0.91535],
[0.27946, 0.03624, 0.03105, 0.94925],
[0.25515, 0.07088, 0.06508, 0.90489],
[0.20376, 0.06648, 0.06437, 0.96139],
[0.24301, 0.03583, 0.03114, 0.98602],
[0.25258, 0.07451, 0.07051, 0.8984],
[0.26794, 0.06236, 0.05757, 0.90813],
[0.25314, 0.08572, 0.08491, 0.87223],
[0.254, 0.05928, 0.04295, 0.93977],
[0.23067, 0.07943, 0.07756, 0.90834],
[0.26383, 0.04378, 0.03587, 0.95252],
[0.28929, 0.02199, 0.01772, 0.967],

[0.24335, 0.11616, 0.10264, 0.83385],
[0.18554, 0.08055, 0.07509, 0.95482],
[0.16838, 0.1049, 0.08313, 0.93959],
[0.27913, 0.13151, 0.11416, 0.7712],
[0.2412, 0.09867, 0.08915, 0.86698],
[0.27267, 0.11465, 0.10446, 0.80422],
[0.2143, 0.08679, 0.07956, 0.91535],
[0.25054, 0.13234, 0.1189, 0.79422],
[0.21776, 0.10816, 0.09702, 0.87306],
[0.28672, 0.10346, 0.08485, 0.82097],
[0.22549, 0.10455, 0.0905, 0.87546],
[0.20235, 0.10456, 0.09489, 0.8942],
[0.28071, 0.10425, 0.09208, 0.81896],
[0.22639, 0.13414, 0.11401, 0.82146],
[0.19805, 0.09128, 0.07901, 0.92766],
[0.1925, 0.09387, 0.08089, 0.92874],
[0.23088, 0.10808, 0.09375, 0.86329],
[0.20948, 0.08627, 0.07999, 0.92026],
[0.27816, 0.13311, 0.1122, 0.77253],
[0.19108, 0.10483, 0.09644, 0.90365],

[0.13861, 0.1029, 0.09256, 0.96193],
[0.22109, 0.14241, 0.14494, 0.78756],
[0.22166, 0.08741, 0.07207, 0.91486],
[0.15366, 0.08405, 0.07434, 0.98395],
[0.12171, 0.10625, 0.10036, 0.96768],
[0.22991, 0.10243, 0.07324, 0.89042],
[0.21233, 0.09575, 0.08309, 0.90483],
[0.21086, 0.11617, 0.08373, 0.88524],
[0.21826, 0.13734, 0.12977, 0.81063],
[0.24111, 0.06866, 0.05934, 0.92689],
[0.24772, 0.11598, 0.11068, 0.82162],
[0.18936, 0.10735, 0.0999, 0.89939],
[0.22125, 0.12862, 0.13171, 0.81442],
[0.16845, 0.13041, 0.11597, 0.88117],
[0.22239, 0.11048, 0.09992, 0.86321],
[0.13759, 0.09407, 0.07819, 0.98615],
[0.19033, 0.12646, 0.12249, 0.85672],
[0.26212, 0.08944, 0.06637, 0.87807],
[0.25172, 0.11854, 0.09875, 0.82699],
[0.21535, 0.10865, 0.09741, 0.87459]])
'''
inputs = np.genfromtxt('SLTrained.csv',delimiter=",")
# output data
#print(inputs)
#values_real_fake = '[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]'
#outputs = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
#outputs = np.genfromtxt('resultTrainedImg.csv',dtype=int, delimiter=",")
#print(outputs)
#outputs.reshape(outputs.size, 1)
#outputs = np.zeros((80, 1))
outputs = np.genfromtxt('SLTrainedResult.csv',dtype=int)
outputs = outputs[:,np.newaxis]
#print(outputs)
# create NeuralNetwork class
class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        self.weights = np.array([[.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50], [.50]])
        self.error_history = []
        self.epoch_list = []

    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # train the neural net for 25,000 iterations
    def train(self, epochs=1000000):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

# create neural network
NN = NeuralNetwork(inputs, outputs)
# train neural network
NN.train()

# create two new examples to predict
# fake examples

tested_values = np.genfromtxt('SLTested.csv',delimiter=",", usecols=(1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12))
tested_files = np.genfromtxt('SLTested.csv',delimiter=",", usecols=(0), dtype=None, encoding=None)
#print(tested_files)
#print(tested_values)
#split_arrays = np.split(tested_values, 40)
#print(split_arrays)

rows = tested_values.shape[0]
cols = tested_values.shape[1]
all_results_arr = np.zeros((40, 2))
#print(rows)
#print(cols)
live_sample = False
tested_files_index = 0
correct_clasify = 0
wrong_clasify = 0
for cols in tested_values:
    #for x in np.nditer(tested_files):
    #print(cols)
        live_percent = 0
        fake_percent = 0

        array_testing = cols[:,np.newaxis]
    #print(array_testing)
        array_transpose = array_testing.T
        #print(x)
        #print(tested_files.item(tested_files_index))
        #print(array_transpose)
        #tested_files_index = tested_files_index + 1
        #print(NN.predict(array_transpose))
        #print(NN.predict(array_transpose), ' - Correct: 0')
        final_prediction = NN.predict(array_transpose)
        final_prediction_str = str(final_prediction)
        final_prediction_str = final_prediction_str[2:-2]
        #print(final_prediction_str)
        file_tested_str = str(tested_files.item(tested_files_index))
        #print(file_tested_str)
        #file_tested_str = file_tested_str[2:-1]
        #print(file_tested_str)
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
        all_results_arr = np.append(all_results_arr, np.array([[file_tested_str, final_prediction_str]]), axis=0)
        #print(all_results_arr)

        if (live_sample == True and "Image" in file_tested_str) or (live_sample == False and "fake" in file_tested_str):
            correct_clasify += 1
        else:
            wrong_clasify += 1

        tested_files_index = tested_files_index + 1

print("Number of correct classifications: " + str(correct_clasify))
print("Number of wrong classifications: " + str(wrong_clasify))
accuracy = (100 * correct_clasify) / (correct_clasify + wrong_clasify)
print("Accuracy: " + str(accuracy) + " %" )

#print("Printing final array")
#print(all_results_arr)

#path_testing = '/home/katerina/Documents/IBP/testing/*'
'''
for cols in all_results_arr:
    #print(cols)
    cols_string = str(cols)
    #print(cols_string)
    if (cols_string == "['0.0' '0.0']"):
        continue
    print(cols_string)

'''


    #img = cv2.imshow(file, 0) # uint8 image in grayscale
'''
example_1 = np.array([[0.19929, 0.06074, 0.0516, 0.98437]]) # fake
example_2 = np.array([[0.22732, 0.07991, 0.06907, 0.9197]]) # fake
example_3 = np.array([[0.1895, 0.10683, 0.09706, 0.90261]])
example_4 = np.array([[0.33486, 0.0453, 0.0408, 0.87504]]) # fake
example_5 = np.array([[0.23366, 0.04772, 0.04447, 0.97015]]) # fake
example_6 = np.array([[0.21144, 0.12056, 0.10873, 0.85527]])
example_7 = np.array([[0.19285, 0.07891, 0.05612, 0.96812]]) # fake
example_8 = np.array([[0.15746, 0.09188, 0.08332, 0.96334]])
example_9 = np.array([[0.28588, 0.03835, 0.03233, 0.93944]]) # fake
example_10 = np.array([[0.21742, 0.04919, 0.03997, 0.98942]]) # fake
example_11 = np.array([[0.18375, 0.10807, 0.09325, 0.91093]])
example_12 = np.array([[0.21797, 0.11987, 0.09488, 0.86328]])
example_13 = np.array([[0.29699, 0.0614, 0.0544, 0.88321]]) # fake
example_14 = np.array([[0.26178, 0.1205, 0.1101, 0.80362]])
example_15 = np.array([[0.21326, 0.05378, 0.04139, 0.98757]]) # fake
example_16 = np.array([[0.2434, 0.12593, 0.10854, 0.81813]])
example_17 = np.array([[0.2158, 0.05406, 0.04914, 0.977]]) # fake
example_18 = np.array([[0.27326, 0.13943, 0.1219, 0.76141]])
example_19 = np.array([[0.27128, 0.0614, 0.05529, 0.90803]]) # fake
example_20 = np.array([[0.2564, 0.10805, 0.09354, 0.83801]])

example_21 = np.array([[0.27115, 0.06842, 0.0644, 0.89203]]) # fake
example_22 = np.array([[0.23885, 0.1148, 0.10436, 0.83799]])
example_23 = np.array([[0.17302, 0.08501, 0.05411, 0.98386]]) # fake
example_24 = np.array([[0.25977, 0.05253, 0.04553, 0.93817]]) # fake
example_25 = np.array([[0.29629, 0.11952, 0.10665, 0.77354]])
example_26 = np.array([[0.26691, 0.09746, 0.08212, 0.84951]])
example_27 = np.array([[0.24977, 0.09903, 0.0863, 0.8609]])
example_28 = np.array([[0.27048, 0.08172, 0.06311, 0.88069]]) # fake
example_29 = np.array([[0.20193, 0.08915, 0.07007, 0.93485]])
example_30 = np.array([[0.27978, 0.10979, 0.09393, 0.8125]])
example_31 = np.array([[0.24912, 0.03937, 0.03281, 0.9747]]) # fake
example_32 = np.array([[0.23466, 0.08414, 0.06965, 0.90755]])
example_33 = np.array([[0.29245, 0.0606, 0.05402, 0.88893]]) # fake
example_34 = np.array([[0.13629, 0.10917, 0.11453, 0.93601]])
example_35 = np.array([[0.22677, 0.05136, 0.04535, 0.97252]]) # fake
example_36 = np.array([[0.2007, 0.06815, 0.05296, 0.97419]]) # fake
example_37 = np.array([[0.2149, 0.13447, 0.12992, 0.81671]])
example_38 = np.array([[0.30335, 0.06607, 0.06109, 0.86549]]) # fake
example_39 = np.array([[0.22447, 0.12966, 0.11234, 0.82953]])
example_40 = np.array([[0.24115, 0.13453, 0.1167, 0.80362]])

# print the predictions for both examples
print(NN.predict(example_1), ' - Correct: 0')
print(NN.predict(example_2), ' - Correct: 0')
print(NN.predict(example_3), ' - Correct: 1')
print(NN.predict(example_4), ' - Correct: 0')
print(NN.predict(example_5), ' - Correct: 0')
print(NN.predict(example_6), ' - Correct: 1')
print(NN.predict(example_7), ' - Correct: 0')
print(NN.predict(example_8), ' - Correct: 1')
print(NN.predict(example_9), ' - Correct: 0')
print(NN.predict(example_10), ' - Correct: 0')
print(NN.predict(example_11), ' - Correct: 1')
print(NN.predict(example_12), ' - Correct: 1')
print(NN.predict(example_13), ' - Correct: 0')
print(NN.predict(example_14), ' - Correct: 1')
print(NN.predict(example_15), ' - Correct: 0')
print(NN.predict(example_16), ' - Correct: 1')
print(NN.predict(example_17), ' - Correct: 0')
print(NN.predict(example_18), ' - Correct: 1')
print(NN.predict(example_19), ' - Correct: 0')
print(NN.predict(example_20), ' - Correct: 1')

print(NN.predict(example_21), ' - Correct: 0')
print(NN.predict(example_22), ' - Correct: 1')
print(NN.predict(example_23), ' - Correct: 0')
print(NN.predict(example_24), ' - Correct: 0')
print(NN.predict(example_25), ' - Correct: 1')
print(NN.predict(example_26), ' - Correct: 1')
print(NN.predict(example_27), ' - Correct: 1')
print(NN.predict(example_28), ' - Correct: 0')
print(NN.predict(example_29), ' - Correct: 1')
print(NN.predict(example_30), ' - Correct: 1')
print(NN.predict(example_31), ' - Correct: 0')
print(NN.predict(example_32), ' - Correct: 1')
print(NN.predict(example_33), ' - Correct: 0')
print(NN.predict(example_34), ' - Correct: 1')
print(NN.predict(example_35), ' - Correct: 0')
print(NN.predict(example_36), ' - Correct: 0')
print(NN.predict(example_37), ' - Correct: 1')
print(NN.predict(example_38), ' - Correct: 0')
print(NN.predict(example_39), ' - Correct: 1')
print(NN.predict(example_40), ' - Correct: 1')
'''
# plot the error over the entire training duration
plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
