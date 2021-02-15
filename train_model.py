"""This script inputs the raw traces and outputs the algorithm performance"""

import trace_file_parser as tp
import oc_svm as oc
import json
import numpy as np
import sys
from os import path
import time

FEATURE_DICT = "ML_FEATURE_DICT_NGRAM.json"
# FEATURE_DICT = 'FEATURE_DICT.json'
FEATURE_VECTOR = {'TF': 0, "TFIDF": 1, "N_GRAM": 2}
RAWTRACE_FILE = {'couchdb_normal': ('raw_tracefile/couchdb_v1_6_normal'),
                 'couchdb_attack': ('raw_tracefile/couchdb_attack_ace'),
                 'mongodb_normal': ('raw_tracefile/mongodb_normal'),
                 'mongodb_attack': ('raw_tracefile/mongodb_brute_force_1', 'raw_tracefile/mongodb_brute_force_2'),
                 'ml': ('raw_tracefile/ml1_normal', 'raw_tracefile/ml2_normal', 'raw_tracefile/ml3_normal',
                       'raw_tracefile/ml4_normal', 'raw_tracefile/ml7_normal')}




INFORMATION_STRING_1 = "# nu, FPR, TPR, std_FPR, std_TPR"
INFORMATION_STRING_2 = "# nu, FPR, TPR"
LEN_FEATURE_VECTOR = 1542 # lenght of ml_ngram

rawtrace_file_normal = RAWTRACE_FILE['mongodb_normal']
rawtrace_file_attack = RAWTRACE_FILE['mongodb_attack'][1]
app_name = 'ml'


# how we sampling real-time streaming system calls
num_syscall_segment =10000
oc_svm_kernel = "rbf"
feature_extraction = "N_GRAM"
feature_extraction_index = FEATURE_VECTOR[feature_extraction]


def dataset_concatenate(dataset_file_list, flag, col_num=LEN_FEATURE_VECTOR):
    """This function combines data from multiple lists into an np array, col_num equals to the number of keys in
    the dist"""
    if isinstance(dataset_file_list, str): # if the input list is a string, then output directly
        dataset = extract_feature_vector(dataset_file_list, FEATURE_DICT, flag)
        return dataset
    else:
        # print("multiple input")
        dataset_total = np.empty((0, col_num))
        for dataset_file in dataset_file_list:
            dataset = extract_feature_vector(dataset_file, FEATURE_DICT, flag)
            # print("dataset: ", dataset.shape)
            dataset_total = np.concatenate((dataset_total, dataset))
        # print("dataset_total: ", dataset_total.shape)
        return dataset_total


def select_with_index(original_list, index_list):
    """ This function allows getting a list with an index list"""
    output_list = []
    for index in index_list:
        output_list.append(original_list[index])
    return output_list


def result_filename(app_name, cross_label="", oc_svm_kernel=oc_svm_kernel, feature_extraction=feature_extraction):
    # return "results/ov-svm/" + app_name + "_ocsvm_"+oc_svm_kernel+'_'+feature_extraction+'.txt'
    return cross_label+app_name + "_ocsvm_"+oc_svm_kernel+'_'+feature_extraction+'.txt'


def extract_feature_vector(rawtrace_file, feature_dict_file, Flag, Read_ngram_dict=True):
    """ parse raw trace and extracts feature vectors.
    INPUT:Flag = 0 for tf, 1 for idf-tf, 2 for n-gram
    OUTPUT: a nested list of feature vectors after normalization"""

    feature_dict = json.load(open(feature_dict_file))

    if Flag == 0:
        feature_vector_list, occurrence_dict, N = tp.parse_trace_tmp(rawtrace_file, feature_dict)
        feature_vector_list = tp.normalization(feature_vector_list)

    if Flag == 1:
        feature_vector_list, occurrence_dict, N = tp.parse_trace_tmp(rawtrace_file, feature_dict)
        feature_vector_list = tp.normalization(feature_vector_list)
        feature_vector_list = tp.df_idf(feature_vector_list, occurrence_dict, N)

    if Flag == 2:
        # if Read_ngram_dict:
            # # Read data from file:
            # feature_dict = json.load(open(FEATURE_DICT))
            # print("length", len(feature_dict))

        # else: # no FEATURE_DICT FOR N-GRAM EXISTS
        #     feature_dict = tp.n_gram_dict(rawtrace_file)
        #     # Serialize data into file:
        #     json.dump(feature_dict, open(NGRAM_FEATURE_DICT, 'w'))

        feature_vector_list = tp.parse_trace_ngram(rawtrace_file, feature_dict)
        feature_vector_list = tp.normalization(feature_vector_list)

    # change into a numpy array for consistence
    feature_vector_list = np.array(feature_vector_list)
    return feature_vector_list


def train_model(filename, oc_svm_kernel=oc_svm_kernel, feature_extraction=feature_extraction):
    """Trace a model with global variable as inputs, currently used for oc-svm, this function applies 10-fold
    cross validation"""
    if_exit = path.exists(filename)
    if not if_exit:
        sys.stdout = open(filename, "w")
        print(INFORMATION_STRING_1)
        feature_extraction_index = FEATURE_VECTOR[feature_extraction]
        print(feature_extraction_index)
        training_set = extract_feature_vector(rawtrace_file_normal, FEATURE_DICT, feature_extraction_index)
        testing_set = extract_feature_vector(rawtrace_file_attack, FEATURE_DICT, feature_extraction_index)
        nu_performance_dict = oc.parameter_search(training_set, testing_set, oc_svm_kernel, oc.nu_list)
        sys.stdout.close()
        time.sleep(1)
    else:
        print("The result file already exists")


def train_model_fv_kernel():
    """ Generate results for all combinations of TF, TF-IDF, gaussian, linear"""
    for fv in ["TF", "TFIDF"]:
        for kernel in ["linear", "rbf"]:
            filename = result_filename(app_name, kernel, fv)
            train_model(filename, kernel, fv)


def cross_parameter_search(training_set_rawtrace, test_set_normal_rawtrace, test_set_attack_rawtrace,
                           kernel=oc_svm_kernel, flag=feature_extraction_index, nu_list=oc.nu_list):
    """This fuction performs cross-app validation"""
    for nu in nu_list:
        training_set = extract_feature_vector(training_set_rawtrace, FEATURE_DICT, flag)
        test_set_normal = extract_feature_vector(test_set_normal_rawtrace, FEATURE_DICT, flag)
        if test_set_attack_rawtrace !=[]:
            test_set_attack = extract_feature_vector(test_set_attack_rawtrace, FEATURE_DICT, flag)
        else:
            test_set_attack = []
            # print("The attack trace is currently not available!")

        FPR, TPR = oc.oc_svm(training_set, test_set_normal, test_set_attack, kernel, nu)
        print(nu, FPR, TPR)


def cross_parameter_search_multiple(training_set_rawtrace, test_set_normal_rawtrace, test_set_attack_rawtrace,
                           kernel=oc_svm_kernel, flag=feature_extraction_index, nu_list=oc.nu_list):
    """This fuction performs cross-app validation for mto1, combine with another function later"""
    for nu in nu_list:
        training_set = dataset_concatenate(training_set_rawtrace, flag)
        test_set_normal = extract_feature_vector(test_set_normal_rawtrace, FEATURE_DICT, flag)
        if test_set_attack_rawtrace !=[]:
            test_set_attack = extract_feature_vector(test_set_attack_rawtrace, FEATURE_DICT, flag)
        else:
            test_set_attack = []
            # print("The attack trace is currently not available!")

        FPR, TPR = oc.oc_svm(training_set, test_set_normal, test_set_attack, kernel, nu)
        print(nu, FPR, TPR)


def one_to_one_cross_validation(filename, training_set_rawtrace_list, test_set_normal_rawtrace_list,
                                kernel=oc_svm_kernel, feature_extraction=feature_extraction):
    """Cross validate the learning algorithm with datas from different apps, current no attack trace for ML"""
    feature_extraction_index = FEATURE_VECTOR[feature_extraction]
    if_exit = path.exists(filename)
    if not if_exit:
        sys.stdout = open(filename, "w")
        print(INFORMATION_STRING_2)
        for i in range(len(training_set_rawtrace_list)):
            for j in range(len(test_set_normal_rawtrace_list)):
                if i == j:
                    continue
                else:
                    training_set_rawtrace = training_set_rawtrace_list[i]
                    test_set_normal_rawtrace = test_set_normal_rawtrace_list[j]
                    print(training_set_rawtrace, test_set_normal_rawtrace)
                    cross_parameter_search(training_set_rawtrace, test_set_normal_rawtrace, [], kernel, feature_extraction_index)
        sys.stdout.close()
        time.sleep(1)
    else:
        print("The result file already exists")


def one_to_one_cv_loop():
    """ Generate results for all combinations of TF, TF-IDF, gaussian, linear"""
    for fv in ["N_GRAM"]:
        for kernel in ["linear", "rbf"]:
            filename = result_filename(app_name, '1to1', kernel, fv)
            one_to_one_cross_validation(filename, RAWTRACE_FILE[app_name], RAWTRACE_FILE[app_name], kernel, fv)


def multiple_to_one_cv_loop():
    """ Generate results for all combinations of TF, TF-IDF, gaussian, linear"""
    for fv in ["N_GRAM"]:
        for kernel in ["linear", "rbf"]:
            filename = result_filename(app_name, 'N21_', kernel, fv)
            multiple_to_one_cross_validation(filename, RAWTRACE_FILE[app_name], kernel, fv)


def multiple_to_one_cross_validation(filename, rawtrace_list, kernel=oc_svm_kernel, feature_extraction=feature_extraction):
    """This function trains the learning algorihtm with multiple training sets and test on one specific data sets
    ;similarly the attack part is missing
    """
    feature_extraction_index = FEATURE_VECTOR[feature_extraction]
    if_exit = path.exists(filename)
    if not if_exit:
        sys.stdout = open(filename, "w")
        print(INFORMATION_STRING_2)
        for i in range(len(rawtrace_list)):
            index_list = list(range(len(rawtrace_list)))
            index_list.remove(i)
            training_set_rawtrace_list = select_with_index(rawtrace_list, index_list)
            testing_set_rawtrace = rawtrace_list[i]
            print('Test: ', testing_set_rawtrace, "Train: ", training_set_rawtrace_list)
            cross_parameter_search_multiple(training_set_rawtrace_list, testing_set_rawtrace, [], kernel, feature_extraction_index)
        sys.stdout.close()
        time.sleep(1)
    else:
        print("The result file already exists")


def main():
    # filename = result_filename(app_name, 'N21_')
    # print(filename)
    # one_to_one_cross_validation(filename, RAWTRACE_FILE[app_name], RAWTRACE_FILE[app_name])

    multiple_to_one_cv_loop()







if __name__ == "__main__":

    main()