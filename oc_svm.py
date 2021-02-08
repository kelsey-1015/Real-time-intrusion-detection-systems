from sklearn.svm import OneClassSVM
from sklearn import metrics
import random
import csv
import numpy as np
from sklearn.model_selection import KFold
import sys


dataset_file_normal = 'couchdb/normal_v1_6_idf.csv'
dataset_file_attack = 'couchdb/attack_v1_6_idf.csv'

# (filename_normal.csv, filename_attack.csv)
dataset_file_list_cb = [('couchdb/cb_normal_tf.csv', 'couchdb/cb_attack_tf.csv'),
                        ('couchdb/cb_normal_tfidf.csv', 'couchdb/cb_attack_tfidf.csv')]

dataset_file_list_mb = [('mongodb/mb_normal_tf.csv', []), ('mongodb/mb_normal_tf.csv', [])]

dataset_file_list_ml_tf = ['ML_algorithm/ml_1_normal_tf.csv', 'ML_algorithm/ml_2_normal_tf.csv',
                           'ML_algorithm/ml_3_normal_tf.csv', 'ML_algorithm/ml_4_normal_tf.csv',
                           "ML_algorithm/ml_7_normal_tf.csv"]


# nu_list = [0.001, 0.005, 0.007, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
nu_list = [0.001, 0.005, 0.007, 0.01, 0.05, 0.1]
gamma_list = ['auto', 'scale']


def select_with_index(original_list, index_list):
    output_list = []
    for index in index_list:
        output_list.append(original_list[index])

    return output_list


def dataset_concatenate(dataset_file_list, col_num = 407):
    """This function combines data from multiple lists into an np array, col_num equals to the number of keys in
    the dist"""
    if isinstance(dataset_file_list, str): # if the input list is a string, then output directly
        dataset = read_data(dataset_file_list)
        return dataset

    fv_list_total = np.empty((0, col_num))
    for dataset_file in dataset_file_list:
        dataset = read_data(dataset_file)
        fv_list_total = np.concatenate((fv_list_total, dataset))
    return fv_list_total



def weighted_by_frequency(feature_vector_list):
    feature_vector_list_n = []
    for feature_vector in feature_vector_list:
        # convert string to int
        feature_vector = list(map(int, feature_vector))
        # weighted by term frequency
        K = sum(feature_vector)
        feature_vector_n = [l/K for l in feature_vector]
        feature_vector_list_n.append(feature_vector_n)

    return feature_vector_list_n


def read_data(csv_file):
    """Read the CSV file and generate datasets as neseted np array"""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        dataset_list = list(reader)

    # dataset_list = weighted_by_frequency(dataset_list)
    dataset_list = np.array(dataset_list)


    return dataset_list


def dataset_split(dataset_list, fraction=0.8):
    """Split the traces, 80% for training, 20% for testing"""

    training_size = round(fraction*len(dataset_list))
    training_set = random.sample(dataset_list, training_size)
    testing_set = [l for l in dataset_list if l not in training_set]
    return training_set, testing_set


def oc_svm(training_set, testing_set_normal, testing_set_attack,  nu_para=0.001, gamma_para='scale'):
    # fit the model
    "rbf: gaussian; linear, Returns -1 for outliers and 1 for inliers."
    clf = OneClassSVM(nu=nu_para, kernel="rbf", gamma=gamma_para)
    clf.fit(training_set)

    if len(testing_set_normal) != 0:
        y_pred_test_normal = clf.predict(testing_set_normal)
        # we expect output 1, so if it predicts a normal sampla as -1, it's an error
        n_error_test_normal = y_pred_test_normal[y_pred_test_normal == -1].size
        FP_rate = n_error_test_normal / len(testing_set_normal)

    else:
        FP_rate = -999


    # using the trained model to detect attacks and compute the True positive rate

    if len(testing_set_attack) != 0:
        y_pred_test_attack = clf.predict(testing_set_attack)
        n_error_test_attack = y_pred_test_attack[y_pred_test_attack == -1].size
        TP_rate = n_error_test_attack / len(testing_set_attack)

    else:
        TP_rate = -999

    return FP_rate, TP_rate


def K_fold(dataset_list_normal, dataset_list_attack, nu, K=10):
    kf = KFold(n_splits=K)
    FPR_list = []
    TPR_list = []
    for train_index, test_index in kf.split(dataset_list_normal):
        train_set, test_set_normal = dataset_list_normal[train_index], dataset_list_normal[test_index]
        FPR, TPR = oc_svm(train_set, test_set_normal, dataset_list_attack, nu)
        FPR_list.append(FPR)
        TPR_list.append(TPR)

    FPR_list = np.array(FPR_list)
    TPR_list = np.array(TPR_list)

    average_FPR = sum(FPR_list)/len(FPR_list)
    average_TPR = sum(TPR_list)/len(TPR_list)

    std_FPR = np.std(FPR_list)
    std_TPR = np.std(TPR_list)

    return average_FPR, average_TPR, std_FPR, std_TPR


def parameter_search(data_list_normal, data_list_attack, nu_list):
    for nu in nu_list:
        FPR, TPR, std_FPR, std_TPR = K_fold(data_list_normal, data_list_attack, nu)
        print(nu, FPR, TPR, std_FPR, std_TPR)


def cross_parameter_search(training_set, test_set_normal, test_set_attack, nu_list):
    for nu in nu_list:
        FPR, TPR = oc_svm(training_set, test_set_normal, test_set_attack, nu)
        print(nu, FPR, TPR)


def cross_parameter_search_loop(training_set_file_list, test_set_normal_file_list, test_set_attack_file_list, nu_list):
    """automate the experiments"""
    # combine traces into one fv
    training_set = dataset_concatenate(training_set_file_list)
    test_set_normal = dataset_concatenate(test_set_normal_file_list)
    test_set_attack = dataset_concatenate(test_set_attack_file_list)
    # print(training_set, test_set_normal, test_set_attack)
    for nu in nu_list:
        FPR, TPR = oc_svm(training_set, test_set_normal, test_set_attack, nu)
        print(nu, FPR, TPR)


def parameter_search_loop(dataset_list):
    """ Automate the experiments"""
    for dataset in dataset_list:
        # print(dataset)
        dataset_file_normal = dataset[0]
        dataset_file_attack = dataset[1]

        if dataset_file_normal !=[]:
            dataset_normal = read_data(dataset_file_normal)
        else:
            dataset_normal = []

        if dataset_file_attack != []:
            dataset_attack = read_data(dataset_file_attack)
        else:
            dataset_attack = []

        parameter_search(dataset_normal, dataset_attack, nu_list)


def one_to_one_cross_validation():
    for i in range(len(dataset_file_list_ml_tf)):
        for j in range(len(dataset_file_list_ml_tf)):
            if i == j:
                continue
            else:
                print('Train: ', i, "test", j)
                training_set_file_list = dataset_file_list_ml_tf[i]
                testing_set_file_list = dataset_file_list_ml_tf[j]
                cross_parameter_search_loop(training_set_file_list, testing_set_file_list, [], nu_list)


def multiple_to_one_cross_validation():

    for i in range(len(dataset_file_list_ml_tf)):
        index_list = list(range(len(dataset_file_list_ml_tf)))
        index_list.remove(i)
        training_set_file_list = select_with_index(dataset_file_list_ml_tf, index_list)
        testing_set_file_list = dataset_file_list_ml_tf[i]

        print('Test: ', testing_set_file_list, "Train: ", training_set_file_list)

        # print('Train: ', i, "test", j)
        # training_set_file_list = dataset_file_list_ml_tf[i]

        cross_parameter_search_loop(training_set_file_list, testing_set_file_list, [], nu_list)

def main():
    """Cross validation ML algorithms (multiple for training, one for testing)"""
    training_set_file_list = dataset_file_list_ml_tf[:-1]
    testing_set_file_list = dataset_file_list_ml_tf[-1]

    """couchdb vs mongodb"""
    # training_set_file_list = 'couchdb/cb_normal_tf.csv'
    # testing_set_file_list = 'mongodb/mb_normal_tf.csv'

    # cross_parameter_search_loop(training_set_file_list, testing_set_file_list, [], nu_list)

    multiple_to_one_cross_validation()
    # one_to_one_cross_validation()

    # print(test.shape)





if __name__ == "__main__":
    main()