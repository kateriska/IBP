import numpy as np

class DecisionTree:
    def __init__(self, trained_vectors, trained_results):
        self.trained_vectors = trained_vectors
        self.trained_results = trained_results
        self.trained_vector_length = np.size(trained_vectors, 1)
        self.number_of_classes = 2
        self.maximum_tree_depth = 1
        self.built_tree = self.processData(trained_vectors, trained_results)

    def processGini(self, trained_vectors, trained_results):
        m = trained_results.size

        if (m <= 1):
            return (None, None)

        fake_count = 0
        live_count = 0
        for result in np.nditer(trained_results):
            #print(item)
            if (result == 0):
                fake_count = fake_count + 1
            elif (result == 1):
                live_count = live_count + 1

        N = np.array([fake_count, live_count])

        gini_sum = 0

        for m_k in N:
            gini_sum = gini_sum + ((m_k / m) ** 2)

        gini = 1.0 - gini_sum

        resulting_treshold = None
        resulting_treshold_index = None

        for trained_vector_index in range(self.trained_vector_length):
            pairs = zip(trained_vectors[:, trained_vector_index], trained_results)
            pairs_tuple = tuple(pairs)
            sorted_tuple = sorted(pairs_tuple, key=lambda pair: pair[0])
            sorted_array = np.array(sorted_tuple)

            all_tresholds_array = sorted_array[:, 0]
            all_classes_array = sorted_array[:, 1]
            all_classes_array = all_classes_array.astype(int)

            m_left = np.array([0,0])
            m_right = N.copy()

            for i in range(1, m):
                previous_class = all_classes_array[i - 1]
                m_left[previous_class] = m_left[previous_class] + 1
                m_right[previous_class] = m_right[previous_class] - 1

                gini_left_sum = 0
                gini_right_sum = 0

                for k in range(self.number_of_classes):
                    gini_left_sum = gini_left_sum + ((m_left[k] / i) ** 2)
                    gini_right_sum = gini_right_sum + ((m_right[k] / (m - i)) ** 2)

                gini_left = 1.0 - gini_left_sum
                gini_right = 1.0 - gini_right_sum

                new_gini = (((i/m)*gini_left) + (((m-i)/m ) * gini_right ))

                if (all_tresholds_array[i] == all_tresholds_array[i - 1]):
                    continue

                if (new_gini < gini):
                    gini = new_gini
                    resulting_treshold_index = trained_vector_index
                    resulting_treshold = (all_tresholds_array[i] + all_tresholds_array[i - 1]) / 2

        return (resulting_treshold, resulting_treshold_index)


    def processData(self, trained_vectors, trained_results, actual_tree_depth=0):
        fake_count = 0
        live_count = 0
        for result in np.nditer(trained_results):
            #print(item)
            if (result == 0):
                fake_count = fake_count + 1
            elif (result == 1):
                live_count = live_count + 1

        vectors_count_each_class = np.array([fake_count, live_count])

        biggest_class = 0
        biggest_class_index = 0
        actual_index = 0

        for class_count in np.nditer(vectors_count_each_class):
            if (class_count > biggest_class):
                biggest_class = class_count
                biggest_class_index = actual_index
            actual_index = actual_index + 1

        tree_node = TreeNode(biggest_class_index)

        if (actual_tree_depth < self.maximum_tree_depth):
            resulting_treshold, resulting_treshold_index = self.processGini(trained_vectors, trained_results)

            if (resulting_treshold_index is not None):
                left_subtree_index = trained_vectors[:, resulting_treshold_index]

                actual_index = 0

                for value in np.nditer(left_subtree_index):
                    if (value < resulting_treshold):
                        left_subtree_index[actual_index] = True
                    else:
                        left_subtree_index[actual_index] = False

                    actual_index = actual_index + 1

                left_subtree_index = left_subtree_index.astype(bool)

                trained_vectors_left_subtree = trained_vectors[left_subtree_index]
                trained_results_left_subtree = trained_results[left_subtree_index]

                trained_vectors_right_subtree = trained_vectors[np.logical_not(left_subtree_index)]
                trained_results_right_subtree = trained_results[np.logical_not(left_subtree_index)]

                tree_node.node_treshold_index = resulting_treshold_index
                tree_node.node_treshold = resulting_treshold
                tree_node.node_left_subtree = self.processData(trained_vectors_left_subtree, trained_results_left_subtree, actual_tree_depth + 1)
                tree_node.node_right_subtree = self.processData(trained_vectors_right_subtree, trained_results_right_subtree, actual_tree_depth + 1)

        return tree_node

    def processTesting(self, tested_vector):
        tree_node = self.built_tree

        while (tree_node.node_left_subtree != None):
            if (tested_vector[tree_node.node_treshold_index] < tree_node.node_treshold):
                tree_node = tree_node.node_left_subtree
            else:
                tree_node = tree_node.node_right_subtree

        return tree_node.node_class_result

class TreeNode:
    def __init__(self, node_class_result):
        self.node_class_result = node_class_result
        self.node_treshold = 0
        self.node_treshold_index = 0
        self.node_left_subtree = None
        self.node_right_subtree = None





trained_vectors = np.genfromtxt('WaveletTrained.csv',delimiter=",")
trained_results = np.genfromtxt('WaveleTrainedResult.csv',dtype=int)

decision_tree = DecisionTree(trained_vectors, trained_results)

tested_vectors = np.genfromtxt('WaveletTested.csv',delimiter=",", usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
tested_files = np.genfromtxt('WaveletTested.csv',delimiter=",", usecols=(0), dtype=None, encoding=None)


rows = tested_vectors.shape[0]
cols = tested_vectors.shape[1]

live_sample = False
tested_files_index = 0
correct_clasify = 0
wrong_clasify = 0
far_value = 0 # the percentage of identification instances in which unauthorised persons are incorrectly accepted
frr_value = 0 # the percentage of identification instances in which authorised persons are incorrectly rejected
for cols in tested_vectors:
        live_percent = 0
        fake_percent = 0

        final_prediction = decision_tree.processTesting(cols)

        final_prediction_str = str(final_prediction)

        final_prediction_int = int(final_prediction_str)
        print(final_prediction_str)

        file_tested_str = str(tested_files.item(tested_files_index))

        if (final_prediction_int == 1):
            live_sample = True
        else:
            live_sample = False
        if (live_sample == True):
            print("File " + file_tested_str + " is classified as LIVE")
        else:
            print("File " + file_tested_str + " is classified as FAKE")


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
