from sklearn.svm import OneClassSVM
from sklearn import metrics
import random
import csv
import numpy as np
from sklearn.model_selection import KFold
import sys

"""TODO:
1. refine cross-validation code
2. automatic select nu with a given criteria"""



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
# FOR TEST
# nu_list =[0.001]

gamma_list = ['auto', 'scale']
LEN_FEATURE_VECTOR = 407


def dataset_concatenate(dataset_file_list, col_num=LEN_FEATURE_VECTOR):
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
    dataset_list = np.array(dataset_list)
    return dataset_list


def oc_svm(training_set, testing_set_normal, testing_set_attack, kernel, nu_para=0.001, gamma_para='scale'):
    """ INPUT: training_set, testing_set_normal, testing_set_attack are nested list of feature vectors"""
    clf = OneClassSVM(nu=nu_para, kernel=kernel, gamma=gamma_para)
    clf.fit(training_set)

    if len(testing_set_normal) != 0:
        y_pred_test_normal = clf.predict(testing_set_normal)
        # we expect output 1, so if it predicts a normal sampla as -1, it's an error
        n_error_test_normal = y_pred_test_normal[y_pred_test_normal == -1].size
        FP_rate = n_error_test_normal / len(testing_set_normal)

    else:
        FP_rate = -999

    if len(testing_set_attack) != 0:
        y_pred_test_attack = clf.predict(testing_set_attack)
        n_error_test_attack = y_pred_test_attack[y_pred_test_attack == -1].size
        TP_rate = n_error_test_attack / len(testing_set_attack)

    else:
        TP_rate = -999
    return FP_rate, TP_rate


def K_fold(dataset_list_normal, dataset_list_attack, kernel, nu, K=10):
    """ This function train and test an oc-svm model with K-fold cross validation"""
    kf = KFold(n_splits=K)
    FPR_list = []
    TPR_list = []
    for train_index, test_index in kf.split(dataset_list_normal):
        train_set, test_set_normal = dataset_list_normal[train_index], dataset_list_normal[test_index]
        FPR, TPR = oc_svm(train_set, test_set_normal, dataset_list_attack, kernel, nu)
        FPR_list.append(FPR)
        TPR_list.append(TPR)

    FPR_list = np.array(FPR_list)
    TPR_list = np.array(TPR_list)

    average_FPR = sum(FPR_list)/len(FPR_list)
    average_TPR = sum(TPR_list)/len(TPR_list)

    std_FPR = np.std(FPR_list)
    std_TPR = np.std(TPR_list)

    return average_FPR, average_TPR, std_FPR, std_TPR


def parameter_search(data_list_normal, data_list_attack, kernel, nu_list):
    """ The function fit the oc-svm model with different parameter nu and outputs the corresponding
    FPR, TPR"""
    nu_performance_dict = {}
    for nu in nu_list:
        FPR, TPR, std_FPR, std_TPR = K_fold(data_list_normal, data_list_attack, kernel, nu)

        print(nu, FPR, TPR, std_FPR, std_TPR)
        nu_performance_dict[nu] = (FPR, TPR, std_FPR, std_TPR)

    return nu_performance_dict


def parameter_search_loop(dataset_list):
    """ Run the parameters search with a list of csv file names"""
    for dataset in dataset_list:
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


def main():
    kernel = 'rbf'

    """Cross validation ML algorithms (multiple for training, one for testing)"""
    # training_set_file_list = dataset_file_list_ml_tf[:-1]
    # testing_set_file_list = dataset_file_list_ml_tf[-1]

    """couchdb vs mongodb"""
    training_set_file_list = 'couchdb/cb_normal_tf.csv'
    testing_set_file_list = 'couchdb/cb_attack_tf.csv'
    # testing_set_file_list = 'mongodb/mb_normal_tf.csv'

    training_set = read_data(training_set_file_list)
    testing_set = read_data(testing_set_file_list)
    nu_performance_dict = parameter_search(training_set, testing_set, kernel, nu_list)


    # multiple_to_one_cross_validation()
    # one_to_one_cross_validation()

    # print(test.shape)





if __name__ == "__main__":
    main()