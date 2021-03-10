"""This script inputs the raw traces and outputs the algorithm performance"""

import trace_file_parser as tp
import oc_svm as oc
import json
import numpy as np
import sys
from os import path
import time
import argparse
from constants import *


FEATURE_VECTOR = {'TF': 0, "TFIDF": 1, "N_GRAM": 2}
INFORMATION_STRING_1 = "# nu, FPR, TPR, std_FPR, std_TPR"
INFORMATION_STRING_2 = "# nu, FPR, TPR"


def dataset_concatenate(rawtrace_file_list, Flag, feature_dict_file, segment_length, filter_flag):
    """This function combines data from multiple lists into an np array, col_num equals to the number of keys in
    the dist"""
    if isinstance(rawtrace_file_list, str): # if the input list is a string, then output directly
        dataset = extract_feature_vector(rawtrace_file_list, feature_dict_file, Flag, segment_length, filter_flag)
        return dataset
    else:
        # print(rawtrace_file_list)
        feature_dict = json.load(open(feature_dict_file))
        col_num = len(feature_dict)+1
        dataset_total = np.empty((0, col_num))
        for rawtrace_file in rawtrace_file_list:
            dataset = extract_feature_vector(rawtrace_file, feature_dict_file, Flag, segment_length, filter_flag)
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


def result_filename(app_name, cross_label, oc_svm_kernel, feature_extraction, dr_flag):
    """return the .txt result output, normally achieve output from stdout"""
    if dr_flag:
        filename = cross_label+app_name + "_ocsvm_"+oc_svm_kernel+'_'+feature_extraction+'_svd.txt'
    else:
        filename = cross_label+app_name + "_ocsvm_"+oc_svm_kernel+'_'+feature_extraction+'txt'
    return filename


def result_labelname(oc_svm_kernel, feature_extraction, dr_flag):
    """Return labels for result output as json file"""
    if dr_flag:
        labelname = oc_svm_kernel+'_'+feature_extraction+'_svd'
    else:
        labelname = oc_svm_kernel+'_'+feature_extraction
    return labelname


def extract_feature_vector(rawtrace_file, feature_dict_file, Flag, segment_length, filter_flag):
    """ parse raw trace and extracts feature vectors.,
    INPUT:Flag = 0 for tf, 1 for idf-tf, 2 for n-gram
    --> segment_length: how the raw tracefile is segmented
    --> filter_flag: whether specific syscalls are filtered out.
    OUTPUT: a nested list of feature vectors after normalization"""

    feature_dict = json.load(open(feature_dict_file))

    if Flag == 0:
        feature_vector_list, occurrence_dict, N = tp.parse_trace_tmp(rawtrace_file, feature_dict, segment_length, filter_flag)
        feature_vector_list = tp.normalization(feature_vector_list)

    if Flag == 1:
        feature_vector_list, occurrence_dict, N = tp.parse_trace_tmp(rawtrace_file, feature_dict, segment_length, filter_flag)
        feature_vector_list = tp.normalization(feature_vector_list)
        feature_vector_list = tp.df_idf(feature_vector_list, occurrence_dict, N)

    if Flag == 2:
        feature_vector_list = tp.parse_trace_ngram(rawtrace_file, feature_dict, segment_length, filter_flag)
        feature_vector_list = tp.normalization(feature_vector_list)

    # change into a numpy array for consistence
    feature_vector_list = np.array(feature_vector_list)
    return feature_vector_list


def cross_parameter_search(training_set_rawtrace, test_set_normal_rawtrace, test_set_attack_rawtrace, feature_dict_file,
                           kernel, flag, nu_list=oc.nu_list):
    """This fuction performs cross-app validation"""
    for nu in nu_list:
        training_set = extract_feature_vector(training_set_rawtrace, feature_dict_file, flag)
        test_set_normal = extract_feature_vector(test_set_normal_rawtrace, feature_dict_file, flag)
        if test_set_attack_rawtrace !=[]:
            test_set_attack = extract_feature_vector(test_set_attack_rawtrace, feature_dict_file, flag)
        else:
            test_set_attack = []
            # print("The attack trace is currently not available!")

        FPR, TPR = oc.oc_svm(training_set, test_set_normal, test_set_attack, kernel, nu)
        print(nu, FPR, TPR)


def cross_parameter_search_multiple(training_set_rawtrace, test_set_normal_rawtrace, test_set_attack_rawtrace,
                                    feature_dict_file, kernel, flag, nu_list=oc.nu_list):
    """This fuction performs cross-app validation for mto1, combine with another function later"""
    for nu in nu_list:
        training_set = dataset_concatenate(training_set_rawtrace, flag, feature_dict_file)
        test_set_normal = extract_feature_vector(test_set_normal_rawtrace, feature_dict_file, flag)
        if test_set_attack_rawtrace !=[]:
            test_set_attack = extract_feature_vector(test_set_attack_rawtrace, feature_dict_file, flag)
        else:
            test_set_attack = []
            # print("The attack trace is currently not available!")

        FPR, TPR = oc.oc_svm(training_set, test_set_normal, test_set_attack, kernel, nu)
        print(nu, FPR, TPR)


def one_to_one_cross_validation(filename, training_set_rawtrace_list, test_set_normal_rawtrace_list,
                                kernel, feature_extraction):
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


def one_to_one_cv_loop(app_name):
    """ Generate results for all combinations of TF, TF-IDF, gaussian, linear"""
    for fv in ["N_GRAM"]:
        for kernel in ["linear", "rbf"]:
            filename = result_filename(app_name, '1to1', kernel, fv)
            one_to_one_cross_validation(filename, RAWTRACE_FILE[app_name], RAWTRACE_FILE[app_name], kernel, fv)


def multiple_to_one_cv_loop(app_name, feature_dict_file):
    """ Generate results for all combinations of TF, TF-IDF, gaussian, linear"""
    for fv in ["TF"]:
        for kernel in ["linear"]:
            filename = result_filename(app_name, 'N21_', kernel, fv)
            multiple_to_one_cross_validation(filename, RAWTRACE_FILE[app_name], feature_dict_file,
                                             kernel, fv)


def multiple_to_one_cross_validation(filename, rawtrace_list, feature_dict_file, kernel,
                                     feature_extraction):
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
            cross_parameter_search_multiple(training_set_rawtrace_list, testing_set_rawtrace, [], feature_dict_file, kernel,
                                            feature_extraction_index)
        sys.stdout.close()
        time.sleep(1)
    else:
        print("The result file already exists")


def train_model(filename, app_name, feature_dict_file, segment_length_list, filter_flag,
                oc_svm_kernel, feature_extraction, dr_flag, dr_dimension):
    """Trace a model with global variable as inputs, currently used for oc-svm, this function applies 10-fold
    cross validation, change output methods to json"""

    rawtrace_file_normal = RAWTRACE_FILE[app_name]['normal']
    if app_name == "ml0":
        rawtrace_file_attack = RAWTRACE_FILE[app_name]['attack'][0]
    else:
        rawtrace_file_attack = RAWTRACE_FILE[app_name]['attack']


    feature_extraction_index = FEATURE_VECTOR[feature_extraction]
    # dict with format {segment_length: {nu: (tpr, fpr, tpr_std, fpr_std}
    segment_dict = {}

    for segment_length in segment_length_list:
        training_set = dataset_concatenate(rawtrace_file_normal, feature_extraction_index, feature_dict_file,
                                               segment_length, filter_flag)
        testing_set = extract_feature_vector(rawtrace_file_attack, feature_dict_file, feature_extraction_index,
                                             segment_length, filter_flag)

        nu_performance_dict = oc.parameter_search(training_set, testing_set, oc_svm_kernel, oc.nu_list,
                                                                  dr_flag, dr_dimension)
        segment_dict[segment_length] = nu_performance_dict

    return segment_dict


def train_model_fv_kernel(app_name, segment_length_list, filter_flag, dr_dimension, dr_flag_list=[True, False],
                          fv_list=["N_GRAM", "TF", "TFIDF"], kernel_list=["rbf", "linear"]):
    """ Generate results for all combinations of TF, TF-IDF, gaussian, linear
    INPUT: dr_flag --> whether perform dimension reduction [truncted SVD]
           dr_dimension --> the number of perform dimension"""
    algorithm_dict = {}
    for fv in fv_list:
        for kernel in kernel_list:
            for dr_flag in dr_flag_list:
                labelname = result_labelname(kernel, fv, dr_flag)
                if fv == "N_GRAM":
                    feature_dict_file = FEATURE_DICT_FILE[fv][app_name]
                else:
                    feature_dict_file = FEATURE_DICT_FILE[fv]
                segment_dict = train_model(labelname, app_name, feature_dict_file, segment_length_list, filter_flag,
                                           kernel, fv, dr_flag, dr_dimension)

                algorithm_dict[labelname] = segment_dict

    return algorithm_dict



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--appname', type=str, default='mongodb', help='input the application name')
    parser.add_argument('--dimension', type=int, default=15, help='input the dimension for trunctedSVD')
    args = parser.parse_args()
    app_name = args.appname
    dr_dimension = args.dimension


    # segment_length_list = [50000]
    # dr_flag_list = [True, False]
    # fv_list = ['TF']
    # kernel_list = ["linear"]
    # filter_flag = False

    segment_length_list = [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 50000]
    dr_flag_list = [True]
    fv_list = ['TF', 'TFIDF', 'N_GRAM']
    kernel_list = ["linear", "rbf"]
    filter_flag = False

    algorithm_dict = train_model_fv_kernel(app_name, segment_length_list, filter_flag, dr_dimension, dr_flag_list,
                                           fv_list, kernel_list)

    json_filename = app_name + "_fpr_ss.json"

    with open(json_filename, "w") as outfile:
        json.dump(algorithm_dict, outfile)


if __name__ == "__main__":

    main()