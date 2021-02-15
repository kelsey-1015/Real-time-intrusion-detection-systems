import csv
from trace_file_parser import *
import json
from itertools import product

input_file_csv = 'syscall_list.csv'
input_file_txt = 'syscall_list.txt'


def ngram_dict(syscall_list, ngram_lenghth=3):
    ngram_list=[]
    for i in product(syscall_list, repeat=ngram_lenghth):
        ngram_list.append(i)
    return ngram_list


def read_data(csv_file):
    """Read the CSV file and generate datasets as neseted np array"""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        dataset_list = list(reader)
    return dataset_list


def parse_file(filename):
    syscall_list = []
    with open(filename) as f:
        for line in f:
            if line.strip():  # skip empty line in the trace file
                syscall_symbol = line.replace(',', '')
                syscall_symbol = syscall_symbol.replace('\n', '')
                syscall_list.append(syscall_symbol)
    return syscall_list


def checking_existing_dict(syscall_list):
    """ Checking if the system call list is complete"""
    index = 0
    for dict in [FEATURE_DICT_1, FEATURE_DICT_2, FEATURE_DICT_3, FEATURE_DICT_4, FEATURE_DICT_5]:
        index += 1
        for k in dict.keys():
            if k not in ['procexit', 'signaldeliver']:
                if k not in syscall_list:
                    print(k, index)


def main():
    syscall_list = parse_file(input_file_txt)
    ngram_list = ngram_dict(syscall_list)
    # # feature_dict = {sys: 0 for sys in syscall_list}
    # # json.dump(feature_dict, open("FEATURE_DICT_NGRAM.json", 'w'))

    print(len(ngram_list))


if __name__ == "__main__":
    main()