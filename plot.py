import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json


def plot_algorithm(data_dict, color_list, linestyle_list, segment_length_list):
    plt.figure()
    index = 0
    for key, list in data_dict.items():
        color_s = color_list[index]
        ls_s = linestyle_list[index]
        plt.plot(segment_length_list, list[0], marker='o', label="TPR_"+key, color=color_s, linestyle=ls_s, linewidth=2)
        plt.plot(segment_length_list, list[1], marker='x', label="FPR_"+key, color=color_s, linestyle=ls_s, linewidth=2)
        index += 1
    plt.legend(prop={'size': 8}, loc='best')
    plt.xlabel("Segment Length (# system call)")
    plt.grid()

    plt.title("oc-svm + truncated SVD with different feature vectors for COUCHDB (mix normal and attack traffic)")
    # plt.title("oc-svm with different feature vectors for COUCHDB (mix normal and attack traffic)")
    # plt.title("oc-svm + truncated SVD with different feature vectors for a ML algorithm")
    # plt.title("oc-svm + truncated SVD with different feature vectors for MONGODB")
    plt.show()


def plot_fpr_reduction(fpr_original_list, fpr_new_list, segment_length_list):
    plt.figure()
    plt.plot(segment_length_list, fpr_original_list, marker='x', label="Original FPR",  linewidth=2)
    plt.plot(segment_length_list, fpr_new_list, marker='x', label="Reduced FPR", linewidth=2)

    plt.legend(prop={'size': 8}, loc='best')
    plt.xlabel("Segment Length (# system call)")
    plt.ylabel("False Positive Rate (FPR)")
    plt.grid()
    plt.title("FPR Reduction for linear-TF of MONGODB for interval = 5")
    plt.show()

def read_data(json_filename):
    with open(json_filename, "r") as read_file:
        dict = json.load(read_file)
    return dict


def data_process(dict, nu):
    """Pre-process the json file so that it can feed into the plot function"""
    dict_reform_total = {}
    for k, v in dict.items():
        tpr_list = []
        fpr_list = []
        segment_list = []
        for k1, v1 in v.items():
            # print(k, k1)
            fpr = v1[nu][0]
            tpr = v1[nu][1]
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            segment_list.append(int(k1))
        dict_reform_total[k] = [tpr_list, fpr_list]
    # split the data to svm and svm_pca
    dict_reform_svd = {}
    dict_reform = {}
    for k, v in dict_reform_total.items():
        ks = k.split("_")
        # print(ks)
        if ks[-1] == 'svd':
            dict_reform_svd[k] = v
        else:
            dict_reform[k] = v
    return dict_reform, dict_reform_svd,  segment_list


def PCA_dimension(data):
    """Plot the variance, used to choose k"""
    pca = PCA().fit(data)
    print(np.cumsum(pca.explained_variance_ratio_))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.show()
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');


def roc_curve(FPR_list, TPR_list):
    plt.figure()
    plt.plot(FPR_list, TPR_list, color='deeppink', linestyle=':', linewidth=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.show()

def bar_plot():
    # labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    labels = ["Gaussian Kernel", "Linear Kernel"]
    exe_time_ngram = [417.68, 490.2537]
    exe_time_tf = [16.85, 17.42]


    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, exe_time_ngram, width, label='N_GRAM')
    rects2 = ax.bar(x + width / 2, exe_time_tf, width, label='TF')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Execution Time (Seconds)')
    ax.set_title('Execution Time for Mongodb with segment length = 30000')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid()
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()

def main():
    # segment_length_list = [2000, 5000, 10000, 15000, 20000, 25000, 30000, 50000]

    # json_filename = "json_result/couchdb_mix.json"
    # output_dict = read_data(json_filename)
    # color_list = ['b', 'g', 'r', 'y', 'k', 'deeppink']
    # line_style_list = ['-', '--', '-', '-', '-.', ':']
    #
    # for nu in [0.01]:
    #     nu = str(nu)
    #
    #     dict_reform, dict_reform_svd, segment_length_list = data_process(output_dict, nu)
    #
    #     # print(len(dict_reform.items()))
    #     # print(dict_reform)
    #     plot(dict_reform_svd, color_list, line_style_list, segment_length_list)

    # app_name = 'ml0'
    # rawtrace_file_normal = RAWTRACE_FILE[app_name]['normal']
    # feature_dict_file = FEATURE_DICT_FILE["TF"]
    # feature_vector_list = tm.extract_feature_vector(rawtrace_file_normal, feature_dict_file, 1, 2000, False)
    # PCA_dimension(feature_vector_list)

    bar_plot()


if __name__ == "__main__":

    main()