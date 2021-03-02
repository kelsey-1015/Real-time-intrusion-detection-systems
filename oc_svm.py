
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc
import csv
import numpy as np
from sklearn.model_selection import KFold
import plot
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

dataset_file_normal = 'couchdb/normal_v1_6_idf.csv'
dataset_file_attack = 'couchdb/attack_v1_6_idf.csv'

# (filename_normal.csv, filename_attack.csv)
dataset_file_list_cb = [('couchdb/cb_normal_tf.csv', 'couchdb/cb_attack_tf.csv'),
                        ('couchdb/cb_normal_tfidf.csv', 'couchdb/cb_attack_tfidf.csv')]

dataset_file_list_mb = [('mongodb/mb_normal_tf.csv', []), ('mongodb/mb_normal_tf.csv', [])]

dataset_file_list_ml_tf = ['ML_algorithm/ml_1_normal_tf.csv', 'ML_algorithm/ml_2_normal_tf.csv',
                           'ML_algorithm/ml_3_normal_tf.csv', 'ML_algorithm/ml_4_normal_tf.csv',
                           "ML_algorithm/ml_7_normal_tf.csv"]


nu_list = [0.001, 0.005, 0.007, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
# nu_list = [0.001, 0.005, 0.007, 0.01, 0.05, 0.1]
# FOR TEST
# nu_list = [0.01]
gamma_list = ['auto', 'scale']


def oc_svm_threshold_test(training_set, testing_set_normal, testing_set_attack, threshold, kernel, nu_para, gamma_para='scale'):
    """This function train a classifier and compute the results with a given threshold, this function is used to
    compute the roc curve if we want using k-fold. ps: the thresholds should be set as distinct values of the scores.
    """
    clf = OneClassSVM(nu=nu_para, kernel=kernel, gamma=gamma_para)
    clf.fit(training_set)

    test_set = np.concatenate((testing_set_normal, testing_set_attack))
    score = clf.decision_function(test_set)
    y_true = np.array([1] * len(testing_set_normal) + [-1] * len(testing_set_attack))
    fpr, tpr, thresholds = roc_curve(y_true, score)
    # print(fpr, tpr, threshold)
    print(thresholds)
    plot.roc_curve(fpr, tpr)


def oc_svm_threshold(training_set, testing_set_normal, testing_set_attack, threshold, kernel, nu_para, gamma_para='scale'):
    """This function train a classifier and compute the results with a given threshold
    """
    clf = OneClassSVM(nu=nu_para, kernel=kernel, gamma=gamma_para)
    clf.fit(training_set)

    score_normal = clf.decision_function(testing_set_normal)
    # print(score_normal)
    predict_normal = [1 if v > threshold else -1 for v in score_normal]
    predict_normal =np.array(predict_normal)
    n_error_test_normal = predict_normal[predict_normal == -1].size
    FP_rate = n_error_test_normal / len(testing_set_normal)

    score_attack =clf.decision_function(testing_set_attack)
    predict_attack = [1 if v > threshold else -1 for v in score_attack]
    predict_attack = np.array(predict_attack)
    n_error_test_attack = predict_attack[predict_attack == -1].size
    TP_rate = n_error_test_attack / len(testing_set_attack)
    return FP_rate, TP_rate



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
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        dataset_list = list(reader)
    dataset_list = np.array(dataset_list)
    return dataset_list


def oc_svm(training_set, testing_set_normal, testing_set_attack, kernel, nu_para=0.001, gamma_para='scale'):
    """ This function train an oc-svm classifier and compute FPR and TPR with default threshold
    INPUT: training_set, testing_set_normal, testing_set_attack are nested list of feature vectors"""
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


def pca_ocsvm(training_set, testing_set_normal, testing_set_attack, kernel, nu_para=0.01, dimension=407, gamma_para='scale'):
    """Fit the PCA with training data, using the resulting vectors to testing data"""
    # print(training_set.shape)
    pca = PCA(n_components=dimension)
    pca.fit(training_set)
    training_set_pca = pca.transform(training_set)
    testing_set_normal_pca = pca.transform(testing_set_normal)
    testing_set_attack_pca = pca.transform(testing_set_attack)
    # print(training_set_pca.shape, testing_set_normal_pca.shape, testing_set_attack_pca.shape)
    # print(testing_set_attack, testing_set_attack_pca)
    FPR, TPR = oc_svm(training_set_pca, testing_set_normal_pca, testing_set_attack_pca, kernel, nu_para, gamma_para)
    print(FPR, TPR)


def truckedsvd_ocsvm(training_set, testing_set_normal, testing_set_attack, kernel, nu_para=0.01, dimension=20, gamma_para='scale'):
    """Fit the PCA with training data, using the resulting vectors to testing data"""
    svd = TruncatedSVD(n_components=dimension)
    svd.fit(training_set)
    training_set_pca = svd.transform(training_set)
    testing_set_normal_svd = svd.transform(testing_set_normal)
    testing_set_attack_svd = svd.transform(testing_set_attack)
    FPR, TPR = oc_svm(training_set_pca, testing_set_normal_svd, testing_set_attack_svd, kernel, nu_para, gamma_para)
    return FPR, TPR




def K_fold(dataset_list_normal, dataset_list_attack, kernel, nu, dr_flag, dr_dimension, K=10):
    """ This function train and test an oc-svm model with K-fold cross validation"""
    if len(dataset_list_normal) != 0:
        kf = KFold(n_splits=K)
    else:
        raise ValueError("The input training data is empty!")
    FPR_list = []
    TPR_list = []
    for train_index, test_index in kf.split(dataset_list_normal):
        train_set, test_set_normal = dataset_list_normal[train_index], dataset_list_normal[test_index]
        if dr_flag:
            FPR, TPR = truckedsvd_ocsvm(train_set, test_set_normal, dataset_list_attack, kernel, nu, dr_dimension)
        else:
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


def parameter_search(data_list_normal, data_list_attack, kernel, nu_list, dr_flag, dr_dimension):
    """ The function fit the oc-svm model with different parameter nu and outputs the corresponding
    FPR, TPR"""
    nu_performance_dict = {}
    for nu in nu_list:
        FPR, TPR, std_FPR, std_TPR = K_fold(data_list_normal, data_list_attack, kernel, nu, dr_flag, dr_dimension)
        nu_performance_dict[nu] = (FPR, TPR, std_FPR, std_TPR)
        print(nu, TPR, FPR)
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
    dataset_file_normal = 'couchdb/cb_normal_tf.csv'
    dataset_file_attack = 'couchdb/cb_attack_tf.csv'
    dataset_normal = read_data(dataset_file_normal)
    dataset_attack = read_data(dataset_file_attack)




if __name__ == "__main__":
    main()