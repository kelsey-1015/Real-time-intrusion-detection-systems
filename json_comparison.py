import json


def compare_json(json_file_1, json_file_2):
    """Compare if two json files are identical"""
    dict_1 = json.load(open(json_file_1))
    dict_2 = json.load(open(json_file_2))
    print(dict_1 == dict_2)


def print_json(json_file, key_list):
    """Print values of a json output according to a key_list; key_list = [algorithm_name, segment_length, nu_value]"""
    dict = json.load(open(json_file))
    value = dict[key_list[0]][key_list[1]][key_list[2]]
    print(value[-1])


def main():
    # json_file_1 = "mongodb_benchmark.json"
    json_file_2 = "test_1.json"
    key_list = ["linear_TF_svd", "50000", "0.01"]
    # r1 = print_json(json_file_1, key_list)
    r2 = print_json(json_file_2, key_list)
    # print(r1==r2)



    # json_file_1 = "mongodb_benchmark.json"
    # json_file_2 = "test_1.json"
    # compare_json(json_file_1, json_file_2)



if __name__ == "__main__":

    main()
