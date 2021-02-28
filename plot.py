import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import oc_svm as oc


def PCA_dimension(data):
    pca = PCA().fit(data)
    print(np.cumsum(pca.explained_variance_ratio_))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.show()
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');


def roc_curve(FPR_list, TPR_list):
    plt.figure()
    plt.plot(FPR_list,TPR_list, color='deeppink', linestyle=':', linewidth=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.show()

def main():

    segment_length_list = [2000, 5000, 10000, 15000, 20000, 25000, 30000, 50000]

    dr_flag_list = [True, False]
    fv_list = ['TF', 'TFIDF', 'N_GRAM']
    kernel_list = ["linear", "rbf"]

    for fv in fv_list:
        for kernel in kernel_list:
            for dr_flag in dr_flag_list:
                pass


if __name__ == "__main__":

    main()