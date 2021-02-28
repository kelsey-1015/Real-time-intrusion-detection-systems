import numpy as np
import matplotlib.pyplot as plt
import json

# couchdb_ocsvm_linear_TF.txt
tpr_list2 = [1, 1, 1, 1, 1, 1, 1, 1]
fpr_list2 = [0.01093, 0.01099, 0.011721, 0.016852, 0.018551, 0.020761, 0.024786, 0.019573]
# couchdb_ocsvm_rbf_TF.txt
tpr_list3 = [1, 1, 1, 1, 1, 1, 1, 1]
fpr_list3 = [0.0138671875, 0.013666, 0.02538, 0.03142, 0.03793, 0.042697, 0.053708, 0.070731]
#couchdb_ocsvm_linear_TFIDF.txt ( 2000 is missing)
tpr_list5 = [0.24736, 0.53157, 0.99444, 1, 1, 1, 1, 1]
fpr_list5 = [0.539257, 0.02077, 0.78145, 0.83138, 0.80968, 0.79632, 0.85456, 0.93170]
#couchdb_ocsvm_rbf_TFIDF.txt ( 2000 is missing)
tpr_list7 = [0.8421, 0.84210, 1, 1, 1, 1, 1, 1]
fpr_list7 = [0.014355, 0.02049, 0.02733, 0.030678, 0.047639, 0.05124, 0.06982, 0.060975]
#couchdb_ocsvm_linear_N_GRAM.txt
tpr_list9 = [0.95, 1, 1, 1, 1, 1, 1, 1]
fpr_list9 = [0.015625, 0.017331981632774766, 0.02050932568149211, 0.0241305, 0.027241,
             0.03422, 0.027621, 0.022073]
#couchdb_ocsvm_rbf_N_GRAM.txt
tpr_list11 = [1, 1, 1, 1, 1, 1, 1, 1]
fpr_list11 = [0.014257, 0.020984, 0.028796, 0.035074, 0.034018, 0.047576, 0.052259, 0.065915]


# couchdb_ocsvm_linear_TF_svd.txt
tpr_list1 = [1, 1, 1, 1, 1, 1, 1, 1]
fpr_list1 = [0.010937, 0.010991, 0.01172, 0.016852, 0.018551, 0.020761, 0.024786, 0.01957]
# couchdb_ocsvm_rbf_TF_svd.txt
tpr_list0 = [0.95, 1, 1, 1, 1, 1, 1, 1]
fpr_list0 = [0.014257, 0.014154, 0.02537, 0.03142, 0.03695, 0.04147, 0.053708, 0.070731]
# couchdb_ocsvm_linear_TFIDF_svd.txt ( 2000 is missing)
tpr_list4 = [0.3999, 0.53158, 0.99444, 1, 1, 1, 1, 1]
fpr_list4 = [0.7125, 0.02077, 0.78145, 0.83138, 0.80968, 0.79632, 0.85456, 0.931707]
#couchdb_ocsvm_rbf_TFIDF_svd.txt( 2000 is missing)
tpr_list6 = [0.57894, 0.42105, 0.85555, 0.9000, 0.90588, 1, 1, 1]
fpr_list6 = [0.013867, 0.019761, 0.0263582, 0.02702, 0.042785, 0.050030, 0.07127, 0.060975]
#couchdb_ocsvm_linear_N_GRAM_svd.txt
tpr_list8 = [0.95, 1, 1, 1, 1, 1, 1, 1]
fpr_list8 = [0.015917, 0.017331, 0.020509, 0.023395, 0.027241, 0.034222, 0.027621, 0.022073]
#couchdb_ocsvm_rbf_N_GRAM_svd.txt
tpr_list10 = [0.95, 1, 1, 1, 1, 1, 1, 1]
fpr_list10 = [0.01367, 0.019765, 0.031721, 0.035804, 0.034019, 0.046356, 0.0522591, 0.068353]

segment_length_list = [2000, 5000, 10000, 15000, 20000, 25000, 30000, 50000]

def plot(data_dict, color_list):
    plt.figure()
    index = 0
    for key, list in data_dict.items():
        color_s = color_list[index]
        plt.plot(segment_length_list, list[0], marker='o', label="TPR_"+key, color=color_s)
        plt.plot(segment_length_list, list[1], marker='x', label="FPR_"+key, color=color_s)
        index += 1
    plt.legend(prop={'size': 10})
    plt.xlabel("Segment Length (# system call)")
    plt.grid()
    plt.title("oc-svm + truncked svd with different feature vectors for couchdb")
    plt.show()


def read_data():
    with open("sample.json", "r") as read_file:
        dict = json.load(read_file)
    return dict

def data_process(dict):
    nu = str(0.01)
    dict_reform = {}
    for k, v in dict.items():
        tpr_list = []
        fpr_list = []
        for k1, v1 in v.items():
            # print(k, k1, v1[nu])
            fpr = v1[nu][0]
            tpr = v1[nu][1]
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        print(fpr_list, tpr_list)
        dict_reform[k]=[tpr_list, fpr_list]
    print(dict_reform)

def main():
    # label_list = ['rbf_TF', 'rbf_TFIDF', 'rbf_NGRAM', 'linear_TF', 'linear_TFIDF', 'linear_NGRAM']
    # data_list = {'linear_TF': [tpr_list2, fpr_list2], 'rbf_TF': [tpr_list3, fpr_list3],
    #              'linear_TFIDF': [tpr_list5, fpr_list5], 'rbf_TFIDF': [tpr_list7, fpr_list7],
    #              'linear_NGRAM': [tpr_list9, fpr_list9], 'rbf_NGRAM': [tpr_list11, fpr_list11],
    #              'linear_TF_svd': [tpr_list1, fpr_list1], 'rbf_TF_svd': [tpr_list0, fpr_list0],
    #              'linear_TFIDF_svd': [tpr_list4, fpr_list4], 'rbf_TFIDF_svd': [tpr_list6, fpr_list6],
    #              'linear_NGRAM_svd': [tpr_list8, fpr_list8], 'rbf_NGRAM_svd': [tpr_list10, fpr_list10]}
    data_list_1 = {'linear_TF': [tpr_list2, fpr_list2], 'rbf_TF': [tpr_list3, fpr_list3],
                 'linear_TFIDF': [tpr_list5, fpr_list5], 'rbf_TFIDF': [tpr_list7, fpr_list7],
                 'linear_NGRAM': [tpr_list9, fpr_list9], 'rbf_NGRAM': [tpr_list11, fpr_list11]}
    data_list_2 = {'linear_TF_svd': [tpr_list1, fpr_list1], 'rbf_TF_svd': [tpr_list0, fpr_list0],
                 'linear_TFIDF_svd': [tpr_list4, fpr_list4], 'rbf_TFIDF_svd': [tpr_list6, fpr_list6],
                 'linear_NGRAM_svd': [tpr_list8, fpr_list8], 'rbf_NGRAM_svd': [tpr_list10, fpr_list10]}
    color_list = ['b', 'g', 'r', 'y', 'k', 'deeppink']
    result_dict = read_data()
    print(result_dict)
    data_process(result_dict)
    # plot(data_list_2, color_list)

if __name__ == "__main__":

    main()
