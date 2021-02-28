import numpy as np
import matplotlib.pyplot as plt

# couchdb_ocsvm_linear_TF.txt
tpr_list2 = [1, 1, 1, 1, 1, 1, 1, 1]
fpr_list2 = [0.015318, 0.01496, 0.01754, 0.02036, 0.016145, 0.024138, 0.020833, 0.02143]
# couchdb_ocsvm_rbf_TF.txt
tpr_list3 = [1, 1, 1, 1, 1, 1, 1, 1]
fpr_list3 = [0.019475, 0.018964, 0.016090, 0.022122, 0.0291607, 0.0267816, 0.020166, 0.040476]
#couchdb_ocsvm_linear_TFIDF.txt ( 2000 is missing)
tpr_list5 = [0.24736, 0.88521, 0.84852, 0.988842, 0.99747, 1, 1, 1]
fpr_list5 = [0.53925, 0.05472, 0.14616, 0.85146, 0.928520, 0.94000, 0.931833, 0.96666]
#couchdb_ocsvm_rbf_TFIDF.txt ( 2000 is missing)
tpr_list7 = [0.42789, 0.92945, 1, 1, 1, 1, 1]
fpr_list7 = [0.027794, 0.02949, 0.036326, 0.037339, 0.033678, 0.03233, 0.040476]
#couchdb_ocsvm_linear_N_GRAM.txt
tpr_list9 = [1, 1, 1, 1, 1, 1, 1, 1]
fpr_list9 = [0.014513, 0.015431, 0.010810, 0.024449, 0.018918, 0.027586, 0.03333, 0.028571]
#couchdb_ocsvm_rbf_N_GRAM.txt
tpr_list11 = [1, 1, 1, 1, 1, 1, 1, 1]
fpr_list11 = [0.024415, 0.028067,0.0321261, 0.056204, 0.050284, 0.050344, 0.0365, 0.033809]


# couchdb_ocsvm_linear_TF_svd.txt
tpr_list1 = [1, 1, 1, 1, 1, 1, 1, 1]
fpr_list1 = [0.015319, 0.014961, 0.017549, 0.020367, 0.016145, 0.024137, 0.020833, 0.021428]
# couchdb_ocsvm_rbf_TF_svd.txt
tpr_list0 = [1, 1, 1, 1, 1, 1, 1, 1]
fpr_list0 = [0.019475, 0.018965, 0.016090, 0.022122, 0.029160, 0.0267816, 0.020167, 0.027142]
# couchdb_ocsvm_linear_TFIDF_svd.txt ( 2000 is missing)
tpr_list4 = []
fpr_list4 = []
#couchdb_ocsvm_rbf_TFIDF_svd.txt( 2000 is missing)
tpr_list6 = []
fpr_list6 = []
#couchdb_ocsvm_linear_N_GRAM_svd.txt
tpr_list8 = [1, 1, 1, 1, 1, 1, 1, 1]
fpr_list8 = [0.014513, 0.015432, 0.010811, 0.0244489, 0.018918, 0.0275862, 0.03333, 0.028571]
#couchdb_ocsvm_rbf_N_GRAM_svd.txt
tpr_list10 = [1, 1, 1, 1, 1, 1, 1, 1]
fpr_list10 = [0.024414, 0.026724, 0.0307927, 0.056204, 0.050284, 0.046897, 0.040666, 0.03380]

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
    plt.title("oc-svm with different feature vectors for MONGODB")
    plt.show()


def main():
    # label_list = ['rbf_TF', 'rbf_TFIDF', 'rbf_NGRAM', 'linear_TF', 'linear_TFIDF', 'linear_NGRAM']
    # data_list = {'linear_TF': [tpr_list2, fpr_list2], 'rbf_TF': [tpr_list3, fpr_list3],
    #              'linear_TFIDF': [tpr_list5, fpr_list5], 'rbf_TFIDF': [tpr_list7, fpr_list7],
    #              'linear_NGRAM': [tpr_list9, fpr_list9], 'rbf_NGRAM': [tpr_list11, fpr_list11],
    #              'linear_TF_svd': [tpr_list1, fpr_list1], 'rbf_TF_svd': [tpr_list0, fpr_list0],
    #              'linear_TFIDF_svd': [tpr_list4, fpr_list4], 'rbf_TFIDF_svd': [tpr_list6, fpr_list6],
    #              'linear_NGRAM_svd': [tpr_list8, fpr_list8], 'rbf_NGRAM_svd': [tpr_list10, fpr_list10]}
    data_list_1 = {'linear_TF': [tpr_list2, fpr_list2], 'rbf_TF': [tpr_list3, fpr_list3],
                 'linear_NGRAM': [tpr_list9, fpr_list9], 'rbf_NGRAM': [tpr_list11, fpr_list11]}
    data_list_2 = {'linear_TF_svd': [tpr_list1, fpr_list1], 'rbf_TF_svd': [tpr_list0, fpr_list0],
                 'linear_NGRAM_svd': [tpr_list8, fpr_list8], 'rbf_NGRAM_svd': [tpr_list10, fpr_list10]}
    color_list = ['b', 'g', 'r', 'y']

    plot(data_list_1, color_list)

if __name__ == "__main__":

    main()
