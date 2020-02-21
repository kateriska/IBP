"""Implementation of the CART algorithm to train decision tree classifiers."""
import numpy as np


class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        #self.n_classes_ = len(set(y))
        self.n_classes_ = 2
        print(self.n_classes_)
        self.n_features_ = np.size(X, 1)
        print(self.n_features_)
        self.tree_ = self._grow_tree(X, y)
    '''
    def predict(self, X):
        for inputs in X:
            prediction = self._predict(inputs)
        #return [self._predict(inputs) for inputs in X]
        return prediction
    '''
    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini_sum = 0
        for n in num_parent:
            best_gini_sum = best_gini_sum + ((n / m) ** 2)
        #best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_gini = 1.0 - best_gini_sum
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] = num_left[c] + 1
                num_right[c] = num_right[c] - 1

                gini_left_sum = 0
                gini_right_sum = 0

                for j in range(self.n_classes_):
                    gini_left_sum = gini_left_sum + ((num_left[j] / i) ** 2)
                    gini_right_sum = gini_right_sum + ((num_right[j] / (m - i)) ** 2)

                gini_left = 1.0 - gini_left_sum
                gini_right = 1.0 - gini_right_sum
                #gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                #gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))
                #gini = (i * gini_left + (m - i) * gini_right) / m
                gini = (((i/m)*gini_left) + (((m-i)/m ) * gini_right ))
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        print("Print nclasses")
        print(self.n_classes_)
        print("Print y")
        print(y)
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        print("Sum per class")
        print(num_samples_per_class)

        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left = X[indices_left]
                y_left = y[indices_left]
                #X_right = X[~indices_left]
                #y_right = y[~indices_left]
                X_right = X[np.logical_not(indices_left)]
                y_right = y[np.logical_not(indices_left)]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


if __name__ == "__main__":
    import sys
    #from sklearn.datasets import load_iris

    #dataset = load_iris()
    trained_vectors = np.genfromtxt('WaveletTrained.csv',delimiter=",")
    trained_results = np.genfromtxt('WaveleTrainedResult.csv',dtype=int)
    X, y = trained_vectors, trained_results # pylint: disable=no-member
    #print(X)
    #print(y)
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y)
    tested_vectors = np.genfromtxt('WaveletTested.csv',delimiter=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
    #print(tested_vectors)
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
            #print(cols)
            #array_testing = cols[:,np.newaxis]
            #print(array_testing)
            #array_transpose = array_testing.T
            #print(array_transpose)

            #final_prediction = clf.predict(array_transpose)
            #for inputs in array_transpose:
                #print(inputs)
            final_prediction = clf._predict(cols)
            print(final_prediction)
            final_prediction_str = str(final_prediction)
            #final_prediction_str = final_prediction_str[1:-1]
            final_prediction_int = int(final_prediction_str)
            print(final_prediction_str)

            file_tested_str = str(tested_files.item(tested_files_index))

            if (final_prediction_int == 1):
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
    #print(clf.predict([[0, 0, 5, 1.5]]))
