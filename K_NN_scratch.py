import math
import ast
import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from trace_file_parser import feature_vector_csv_generator

dataset_file_normal_tf = 'normal_v1_6_tf.csv'
dataset_file_attack_tf = 'attack_v1_6_tf.csv'

dataset_file_normal_idf = 'normal_v1_6_idf.csv'
dataset_file_attack_idf = 'attack_v1_6_idf.csv'

dataset_file_normal_ngram = 'normal_v1_6_ngram_norm.csv'
dataset_file_attack_ngram = 'attack_v1_6_ngram_norm.csv'


def dataset_split(dataset_list, fraction=0.9):
    """Split the traces, get the same random items out of the list every time"""
    # print(dataset_list)
    training_size = round(fraction*len(dataset_list))
    random.seed(training_size)
    index_list = range(len(dataset_list))
    index_list_training = random.sample(index_list, training_size)
    index_list_test = [l for l in index_list if l not in index_list_training]
    training_set = [dataset_list[i] for i in index_list_training]
    # print(training_set)
    testing_set = [dataset_list[i] for i in index_list_test]
    # for i in index_list_training:
    #     training_set.append(dataset_list[i])

    return training_set, testing_set


def read_data(csv_file):
    """Read the CSV file and generate datasets as neseted np array"""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        dataset_list = list(reader)

    # dataset_list = weighted_by_frequency(dataset_list)
    # dataset_list = np.array(dataset_list)

    # change string to float
    dataset_list_num = []
    for fv in dataset_list:
        fv = [float(c) for c in fv]
        if fv not in dataset_list_num:
            dataset_list_num.append(fv)
    return dataset_list_num


def generate_datasets(fv_list_normal, fv_list_attack):
    """Returns -1 for attacks and 1 for normal traces."""

    training_set, test_set_normal = dataset_split(fv_list_normal)
    label_test_normal = [1] * len(test_set_normal)
    label_test_attack = [-1] * len(fv_list_attack)
    label_test = label_test_normal + label_test_attack
    test_set = test_set_normal + fv_list_attack

    return training_set, test_set, label_test




def euclidean_distance(x, y):

    x = [float(c) for c in x]
    y = [float(c) for c in y]
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return distance


def get_neighbors(training_set, test_sample, num_neighbours=1):
    """ Get the nearest neighbours"""
    distances = []
    for training_sample in training_set:
        dist = euclidean_distance(training_sample, test_sample)
        distances.append((training_sample, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbours = []
    neighbours_dist = []
    for i in range(num_neighbours):
        neighbours.append(distances[i][0])
        neighbours_dist.append(distances[i][1])
        average_dist = sum(neighbours_dist)/len(neighbours_dist)
    return neighbours, average_dist



def KNN_prediction(training_set, test_sample, threshold):
    """returns - 1 for outliers and 1 for inliers."""
    _, dist = get_neighbors(training_set, test_sample)
    # print(dist)
    if dist >= threshold:  # it's an outlier/attack
        predict_label = -1
    else:
        predict_label = 1
    # print(predict_label)

    return predict_label


def dimension_redection(data_set):
    data_set = np.array(data_set)
    pca = PCA(n_components=5)
    pca.fit(data_set)
    data_set = pca.transform(data_set)
    data_set = list(data_set)
    return data_set


def KNN(dataset_list_normal, dataset_list_attack, threshold_list=[0.1]):
    """returns - 1 for outliers and 1 for normal."""
    result_nested_list = []
    training_set, test_set, label_list = generate_datasets(dataset_list_normal, dataset_list_attack)
    TPR_list, FPR_list = [], []

    for thresh in threshold_list:
        result_list = [thresh, 0, 0]
        TN, FN, FP, TP = 0, 0, 0, 0
        for index in range(len(test_set)):
            test_sample = test_set[index]
            pred_label = KNN_prediction(training_set, test_sample, thresh)
            ground_label = label_list[index]
            # print(pred_label, ground_label)
            if pred_label == 1 and ground_label == 1:
                TN +=1
            elif pred_label == -1 and ground_label == 1: # false alarm
                FP += 1
            elif pred_label == 1 and ground_label == -1:
                FN += 1
            elif pred_label == -1 and ground_label == -1: # true alarm
                TP += 1
        FPR = FP/sum([i == 1 for i in label_list])
        TPR = TP/sum([i == -1 for i in label_list])
        TPR_list.append(TPR)
        FPR_list.append(FPR)
        result_list[1] = FPR
        result_list[2] = TPR
        # print(result_list)
        result_nested_list.append(result_list)


    return result_nested_list, TPR_list, FPR_list


def ROC_curve_plot(FPR_1, TPR_1, label_1, FPR_2, TPR_2, label_2, FPR_3, TPR_3, label_3):
    plt.plot(FPR_1, TPR_1, label=label_1)
    plt.plot(FPR_2, TPR_2, label=label_2)
    plt.plot(FPR_3, TPR_3, label=label_3)

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("The ROC curve of PCA and kNN with various feature vectors")
    plt.grid()
    plt.legend()
    plt.show()




def main():

    dataset_list_normal_tf = read_data(dataset_file_normal_tf)
    dataset_list_attack_tf = read_data(dataset_file_attack_tf)
    dataset_list_normal_idf = read_data(dataset_file_normal_idf)
    dataset_list_attack_idf = read_data(dataset_file_attack_idf)
    dataset_list_normal_ngram = read_data(dataset_file_normal_ngram)
    dataset_list_attack_ngram = read_data(dataset_file_attack_ngram)

    data_pca_attack_tf= dimension_redection(dataset_list_attack_tf)
    data_pca_normal_tf = dimension_redection(dataset_list_normal_tf)
    data_pca_attack_idf = dimension_redection(dataset_list_attack_idf)
    data_pca_normal_idf = dimension_redection(dataset_list_normal_idf)
    data_pca_attack_ngram = dimension_redection(dataset_list_attack_ngram)
    data_pca_normal_ngram = dimension_redection(dataset_list_normal_ngram)


    threshold_0 = np.arange(0, 0.2, 0.005)
    threshold_1 = np.arange(0.2, 1.2, 0.2)
    threshold_2 = np.arange(1.2, 10, 1)

    threshold_list_tf = np.concatenate((threshold_0, threshold_1))
    threshold_list_idf = np.concatenate((threshold_0, threshold_1))
    threshold_list_ngram = np.concatenate((threshold_0, threshold_1))


    _, TPR_list_tf, FPR_list_tf = KNN(data_pca_normal_tf, data_pca_attack_tf, threshold_list_tf)

    _, TPR_list_idf, FPR_list_idf = KNN(data_pca_normal_idf, data_pca_attack_idf, threshold_list_idf)
    #
    _, TPR_list_ngram, FPR_list_ngram = KNN(data_pca_normal_ngram, data_pca_attack_ngram, threshold_list_ngram)


    ROC_curve_plot(FPR_list_tf, TPR_list_tf, "tf", FPR_list_idf, TPR_list_idf, "idf", FPR_list_ngram, TPR_list_ngram, "n-gram",)

    print(FPR_list_tf, TPR_list_tf, "tf", FPR_list_idf, TPR_list_idf, "idf", FPR_list_ngram, TPR_list_ngram, "n-gram")

    # print(dist)




if __name__ == "__main__":
    main()