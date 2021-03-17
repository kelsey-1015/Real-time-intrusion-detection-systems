import json


def compare_json(json_file_1, json_file_2):
    """Compare if two json files are identical"""
    dict_1 = json.load(open(json_file_1))
    dict_2 = json.load(open(json_file_2))
    print(dict_1 == dict_2)


def print_json_key(json_file, key_list):
    """Print values of a json output according to a key_list; key_list = [algorithm_name, segment_length, nu_value]"""
    dict = json.load(open(json_file))
    value = dict[key_list[0]][key_list[1]][key_list[2]]
    print(value[-1])


def print_json(json_file):
    dict = json.load(open(json_file))
    # print(dict)
    for k, v in dict.items():
        print(k)

def print_num_key(json_file):
    dict = json.load(open(json_file))
    print("The length of the dict is: ", len(dict.keys()))


def extract_json(input_json_file, output_json_file):
    """Load a jsonfile; Extract [k, v] pair with a specific condition; output into a new json file"""
    dict = json.load(open(input_json_file))
    dict_new = {}
    for k, v in dict.items():
        if 'GRAM' not in k:
            dict_new[k] =v
    with open(output_json_file, "w") as outfile:
        json.dump(dict_new, outfile)


def combine_json(input_json_file_1, input_json_file_2, output_json_file):
    dict_1 = json.load(open(input_json_file_1))
    dict_2 = json.load(open(input_json_file_2))
    dict_merge = {**dict_1, **dict_2}
    print(dict_merge)
    with open(output_json_file, "w") as outfile:
        json.dump(dict_merge, outfile)

def main():

    # key_list = ["rbf_TF", "30000", "0.01"]
    json_file_1 = 'MONGODB_FEATURE_DICT_NGRAM_6.json'
    json_file_2 = 'feature_vector_json/MONGODB_FEATURE_DICT_NGRAM.json'
    print_num_key(json_file_1)









if __name__ == "__main__":

    main()
