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
        plt.plot(segment_length_list, list[0], marker='o', label="TPR_"+key, color=color_s, linestyle=ls_s,
                 markersize=10, linewidth=4)
        plt.plot(segment_length_list, list[1], marker='x', label="FPR_"+key, color=color_s, linestyle=ls_s,
                 markersize=10, linewidth=4)
        index += 1
    plt.legend(prop={'size': 10}, ncol=2, handleheight=2.4, labelspacing=0.1)
    # plt.legend(prop={'size': 10}, loc=0)
    plt.xlabel("Segment Length (# system call)", fontsize=20)
    plt.ylabel("TPR/FPR", fontsize=20)
    plt.grid()
    # plt.title("TPR and FPR with different segment window size for COUCHDB", fontsize=16)
    plt.show()


# def plot_fpr_reduction(fpr_original_list, fpr_new_list, segment_length_list):
#     plt.figure()
#     plt.plot(segment_length_list, fpr_original_list, marker='x', label="Original FPR",  linewidth=2)
#     plt.plot(segment_length_list, fpr_new_list, marker='x', label="Reduced FPR", linewidth=2)
#
#     plt.legend(prop={'size': 8}, loc='best')
#     plt.xlabel("Segment Length (# system call)")
#     plt.ylabel("False Positive Rate (FPR)")
#     plt.grid()
#     plt.title("FPR Reduction for linear-TF of MONGODB for interval = 5")
#     plt.show()


def plot_fpr_reduction(fpr_original_list_1, fpr_new_list_1,
                       fpr_original_list_2, fpr_new_list_2,segment_length_list):

    fig, axs = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')

    axs[0].plot(segment_length_list, fpr_original_list_1, marker='x', color='red', label="Original FPR for TF",  linewidth=2)
    axs[0].plot(segment_length_list, fpr_new_list_1, marker='x', color='blue', label="Reduced FPR for TF", linewidth=2)
    axs[1].plot(segment_length_list, fpr_original_list_2, marker='x', color='red', label="Original FPR for NGRAM", linewidth=2)
    axs[1].plot(segment_length_list, fpr_new_list_2, marker='x', color='blue', label="Reduced FPR for NGRAM", linewidth=2)

    axs[0].legend(prop={'size': 8}, loc='best')
    axs[1].legend(prop={'size': 8}, loc='best')
    axs[0].set_xlabel("Segment Length (# system call)")
    axs[1].set_xlabel("Segment Length (# system call)")
    axs[0].set_ylabel("False Positive Rate (FPR)")
    # axs[1].set_ylabel("False Positive Rate (FPR)")
    axs[0].set_ylim([-0.005, 0.035])
    axs[1].set_ylim([-0.005, 0.035])
    axs[0].grid()
    axs[1].grid()


    # plt.title("FPR Reduction for linear-TF of MONGODB for interval = 5")
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
    labels = ['MongoDB', 'CouchDB', "Image Classification"]
    # labels = ["Gaussian Kernel", "Linear Kernel"]
    exe_time_ngram_linear = [490.3, 330.1, 15.8]
    exe_time_ngram_rbf = [417.7, 327.0, 15.7]
    exe_time_tf_linear = [17.4, 74.9, 8.2]
    exe_time_tf_rbf = [16.9, 71.3, 8.3]


    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 3*width/2, exe_time_ngram_linear, width, color ='blue', label='N_GRAM_linear')
    rects2 = ax.bar(x - width/2, exe_time_tf_linear, width, color='red', label='TF_linear')
    rects3 = ax.bar(x + width/2, exe_time_ngram_rbf, width, color='blue', label='N_GRAM_rbf', hatch="//")
    rects4 = ax.bar(x + width*3/2, exe_time_tf_rbf, width, color='red', label='TF_rbf', hatch="//")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Execution Time (Seconds)', fontsize=20)
    # ax.set_title('Execution Time for Mongodb with segment length = 30000')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=20)
    ax.grid()
    ax.legend(fontsize=16)

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
    autolabel(rects3)
    autolabel(rects4)

    fig.tight_layout()

    plt.show()

def main():
    # segment_length_list = [2000, 5000, 10000, 15000, 20000, 25000, 30000, 50000]

    # json_filename = "json_result/couchdb_mix.json"
    # output_dict = read_data(json_filename)
    # color_list = ['b', 'g', 'r', 'y', 'k', 'deeppink']
    # line_style_list = ['-', '--', '-', '-', '-.', ':']
    # #
    # for nu in [0.01]:
    #     nu = str(nu)
    #     dict_reform, dict_reform_svd, segment_length_list = data_process(output_dict, nu)
    #     plot_algorithm(dict_reform, color_list, line_style_list, segment_length_list)



    # app_name = 'ml0'
    # rawtrace_file_normal = RAWTRACE_FILE[app_name]['normal']
    # feature_dict_file = FEATURE_DICT_FILE["TF"]
    # feature_vector_list = tm.extract_feature_vector(rawtrace_file_normal, feature_dict_file, 1, 2000, False)
    # PCA_dimension(feature_vector_list)



    # bar_plot()
    plot_fpr_reduction()


if __name__ == "__main__":

    main()