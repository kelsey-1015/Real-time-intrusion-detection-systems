RAWTRACE_FILE = {'couchdb': {'normal': ('raw_tracefile/couchdb_normal_11', 'raw_tracefile/couchdb_normal_12',
                                        'raw_tracefile/couchdb_normal_13', 'raw_tracefile/couchdb_normal_14',
                                                                           'raw_tracefile/couchdb_v1_6_normal'),
                             'attack': 'raw_tracefile/couchdb_attack_mix'},
                 'mongodb': {'normal': 'raw_tracefile/mongodb_normal', 'attack': 'raw_tracefile/mongodb_brute_force_2'},
                 'ml0': {'normal': 'raw_tracefile/ml0_normal', 'attack': ('raw_tracefile/ml0_attack',
                                                                          'raw_tracefile/ml0_attack_1')},
                 'ml': ('raw_tracefile/ml1_normal', 'raw_tracefile/ml2_normal', 'raw_tracefile/ml3_normal',
                       'raw_tracefile/ml4_normal', 'raw_tracefile/ml7_normal')}


FEATURE_DICT_FILE = {'TF': "feature_vector_json/FEATURE_DICT.json", "TFIDF": "feature_vector_json/FEATURE_DICT.json",
                     "N_GRAM": {'couchdb': 'feature_vector_json/COUCHDB_FEATURE_DICT_NGRAM.json',
                                'mongodb': 'feature_vector_json/MONGODB_FEATURE_DICT_NGRAM.json',
                                 'ml0': 'feature_vector_json/ML0_FEATURE_DICT_NGRAM.json'}}


FEATURE_VECTOR = {'TF': 0, "TFIDF": 1, "N_GRAM": 2}
INFORMATION_STRING_1 = "# nu, FPR, TPR, std_FPR, std_TPR"
INFORMATION_STRING_2 = "# nu, FPR, TPR"

# The number of normal test data for mongodb application
SL_TN_number_mongodb = {1000: 746, 2000: 373, 5000: 149, 10000: 75, 15000: 50, 20000:37, 25000: 30, 30000: 25, 50000: 15}

"""The settings used to generate mongodb_benchmark"""
# segment_length_list = [20000, 50000]
# dr_flag_list = [True, False]
# fv_list = ['TF', 'TFIDF', 'N_GRAM']
# kernel_list = ["linear"]
# filter_flag = False

"""The settings for normal test"""
# segment_length_list = [50000]
# dr_flag_list = [True, False]
# fv_list = ['TF']
# kernel_list = ["linear"]
# filter_flag = False

"""The full settings"""
# segment_length_list = [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 50000]
# dr_flag_list = [True, False]
# fv_list = ['TF', 'TFIDF', 'N_GRAM']
# kernel_list = ["linear", "rbf"]
# filter_flag = False
