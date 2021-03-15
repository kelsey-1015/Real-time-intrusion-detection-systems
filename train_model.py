"""This script inputs the raw traces and outputs the algorithm performance"""

import trace_file_parser as tp
import oc_svm as oc
import json
import numpy as np
import time
import argparse
from constants import *


def dataset_concatenate(rawtrace_file_list, Flag, feature_dict_file, segment_length, filter_flag):
    """This function combines data from multiple lists into an np array, col_num equals to the number of keys in
    the dist"""
    if isinstance(rawtrace_file_list, str): # if the input list is a string, then output directly
        dataset = extract_feature_vector(rawtrace_file_list, feature_dict_file, Flag, segment_length, filter_flag)
        return dataset
    else:
        feature_dict = json.load(open(feature_dict_file))
        col_num = len(feature_dict)+1
        dataset_total = np.empty((0, col_num))
        for rawtrace_file in rawtrace_file_list:
            dataset = extract_feature_vector(rawtrace_file, feature_dict_file, Flag, segment_length, filter_flag)
            dataset_total = np.concatenate((dataset_total, dataset))
        return dataset_total


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



def train_model(filename, app_name, feature_dict_file, segment_length_list, filter_flag,
                oc_svm_kernel, feature_extraction, dr_flag, dr_dimension):
    """Trace a model with global variable as inputs, currently used for oc-svm, this function applies 10-fold
    cross validation, change output methods to json"""

    rawtrace_file_normal = RAWTRACE_FILE[app_name]['normal']
    if app_name == "ml0":
        rawtrace_file_attack = RAWTRACE_FILE[app_name]['attack'][1]
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


def train_model_fv_kernel(app_name, segment_length_list, filter_flag, dr_dimension, dr_flag_list,
                          fv_list, kernel_list):
    """ Generate results for all combinations of TF, TF-IDF, gaussian, linear
    INPUT: dr_flag --> whether perform dimension reduction [truncted SVD]
           dr_dimension --> the number of perform dimension"""
    algorithm_dict = {}
    execution_time_dict ={}
    for fv in fv_list:
        for kernel in kernel_list:
            for dr_flag in dr_flag_list:
                labelname = result_labelname(kernel, fv, dr_flag)
                print("labelname: ", labelname)
                start_time = time.time()
                if fv == "N_GRAM":
                    feature_dict_file = FEATURE_DICT_FILE[fv][app_name]
                else:
                    feature_dict_file = FEATURE_DICT_FILE[fv]
                segment_dict = train_model(labelname, app_name, feature_dict_file, segment_length_list, filter_flag,
                                           kernel, fv, dr_flag, dr_dimension)
                execution_time = time.time() - start_time
                execution_time_dict[labelname] = execution_time
                algorithm_dict[labelname] = segment_dict

    return algorithm_dict, execution_time_dict



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--appname', type=str, default='couchdb', help='input the application name')
    parser.add_argument('--dimension', type=int, default=15, help='input the dimension for trunctedSVD')
    args = parser.parse_args()
    app_name = args.appname
    dr_dimension = args.dimension

    segment_length_list = [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 50000]
    dr_flag_list = [False, True]
    fv_list = ["N_GRAM"]
    kernel_list = ["linear", "rbf"]
    filter_flag = False

    algorithm_dict, execution_time_dict = train_model_fv_kernel(app_name, segment_length_list, filter_flag,
    dr_dimension, dr_flag_list, fv_list, kernel_list)


    print(algorithm_dict)
    print(execution_time_dict)

    json_filename_execution = app_name + "_execution_time_NGRAM.json"
    json_filename_results = app_name + "_fpr_ss_NGRAM.json"

    with open(json_filename_execution, "w") as outfile:
        json.dump(execution_time_dict, outfile)

    with open(json_filename_results, "w") as outfile:
        json.dump(algorithm_dict, outfile)


if __name__ == "__main__":

    main()