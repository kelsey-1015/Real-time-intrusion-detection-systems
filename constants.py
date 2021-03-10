RAWTRACE_FILE = {'couchdb': {'normal': ('raw_tracefile/couchdb_normal_1', 'raw_tracefile/couchdb_v1_6_normal'),
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
