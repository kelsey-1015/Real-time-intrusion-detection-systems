import os
import csv
import math
import json
import ast
from oc_svm import read_data
import numpy as np


"""TODO LIST
2. make normalization consistent
4. make csv naming strategy consistent"""

NGRAM_LENGTH = 3

# distinct_syscall_normal_trace
FEATURE_DICT_1 = {'futex': 0, 'epoll_ctl': 0, 'write': 0, 'accept': 0, 'epoll_wait': 0, 'timerfd_settime': 0, 'read': 0,
                'sched_yield': 0, 'rt_sigtimedwait': 0, 'wait4': 0, 'select': 0, 'mmap': 0, 'munmap': 0, 'writev': 0,
                'recvfrom': 0, 'close': 0, 'fcntl': 0, 'getsockopt': 0, 'setsockopt': 0, 'getpeername': 0, 'getpid': 0,
                'stat': 0, 'access': 0, 'open': 0, 'fstat': 0, 'lseek': 0, 'pread': 0, 'fsync': 0, 'rename': 0,
                'socket': 0, 'connect': 0, 'poll': 0, 'sendto': 0, 'ioctl': 0, 'bind': 0, 'getsockname': 0, 'times': 0,
                'sysinfo': 0}

# dictinct_syscall_normal_trace_attack1
FEATURE_DICT_2 = {'futex': 0, 'epoll_ctl': 0, 'write': 0, 'accept': 0, 'epoll_wait': 0, 'timerfd_settime': 0, 'read': 0,
                  'sched_yield': 0, 'rt_sigtimedwait': 0, 'wait4': 0, 'select': 0, 'mmap': 0, 'munmap': 0, 'writev': 0,
                  'recvfrom': 0, 'close': 0, 'fcntl': 0, 'getsockopt': 0, 'setsockopt': 0, 'getpeername': 0,
                  'getpid': 0, 'stat': 0, 'access': 0, 'open': 0, 'fstat': 0, 'lseek': 0, 'pread': 0, 'fsync': 0,
                  'rename': 0, 'socket': 0, 'connect': 0, 'poll': 0, 'sendto': 0, 'ioctl': 0, 'bind': 0, 'getsockname': 0,
                  'times': 0, 'sysinfo': 0, 'setsid': 0, 'rt_sigprocmask': 0, 'execve': 0, 'brk': 0, 'mprotect': 0,
                  'arch_prctl': 0, 'rt_sigaction': 0, 'geteuid': 0, 'getppid': 0, 'dup': 0, 'clone': 0, 'kill': 0,
                  'exit_group': 0, 'procexit': 0, 'signaldeliver': 0, 'sigreturn': 0, 'umask': 0, 'lstat': 0, 'newfstatat': 0,
                  'unlinkat': 0, 'pipe': 0, 'vfork': 0}

# dictinct_syscall_normal_trace_attack1_v_6
FEATURE_DICT_3 = {'futex': 0, 'epoll_ctl': 0, 'write': 0, 'accept': 0, 'epoll_wait': 0, 'timerfd_settime': 0, 'read': 0,
                  'sched_yield': 0, 'rt_sigtimedwait': 0, 'wait4': 0, 'select': 0, 'mmap': 0, 'munmap': 0, 'writev': 0,
                  'recvfrom': 0, 'close': 0, 'fcntl': 0, 'getsockopt': 0, 'setsockopt': 0, 'getpeername': 0, 'getpid': 0,
                  'stat': 0, 'access': 0, 'open': 0, 'fstat': 0, 'lseek': 0, 'pread': 0, 'fsync': 0, 'rename': 0, 'socket': 0,
                  'connect': 0, 'poll': 0, 'sendto': 0, 'ioctl': 0, 'bind': 0, 'getsockname': 0, 'times': 0, 'sysinfo': 0,
                  'setsid': 0, 'rt_sigprocmask': 0, 'execve': 0, 'brk': 0, 'mprotect': 0, 'arch_prctl': 0, 'rt_sigaction': 0,
                  'geteuid': 0, 'getppid': 0, 'dup': 0, 'clone': 0, 'kill': 0, 'exit_group': 0, 'procexit': 0, 'signaldeliver': 0,
                  'sigreturn': 0, 'umask': 0, 'lstat': 0, 'newfstatat': 0, 'unlinkat': 0, 'pipe': 0, 'vfork': 0, 'openat': 0,
                  'getdents': 0, 'uname': 0, 'statfs': 0}

# dictinct_syscall_normal_mongodb
FEATURE_DICT_4 = {'futex': 0, 'nanosleep': 0, 'open': 0, 'fstat': 0, 'getdents': 0, 'close': 0, 'getrusage': 0, 'read': 0,
                  'epoll_wait': 0, 'accept': 0, 'epoll_ctl': 0, 'ioctl': 0, 'getsockname': 0, 'setsockopt': 0, 'getsockopt': 0,
                  'getpeername': 0, 'write': 0, 'gettid': 0, 'prctl': 0, 'getrlimit': 0, 'clone': 0, 'set_robust_list': 0,
                  'recvmsg': 0, 'sendmsg': 0, 'writev': 0, 'lstat': 0, 'unlink': 0, 'stat': 0, 'pread': 0, 'fdatasync': 0,
                  'pwrite': 0, 'sched_yield': 0, 'rename': 0, 'mmap': 0, 'mprotect': 0, 'brk': 0, 'ftruncate': 0, 'select': 0,
                  'madvise': 0, 'fallocate': 0, 'shutdown': 0, 'exit': 0, 'procexit': 0}


# mongo+couchdb/normal
FEATURE_DICT_5 = {'futex': 0, 'nanosleep': 0, 'open': 0, 'fstat': 0, 'getdents': 0, 'close': 0, 'getrusage': 0, 'read': 0,
                  'epoll_wait': 0, 'accept': 0, 'epoll_ctl': 0, 'ioctl': 0, 'getsockname': 0, 'setsockopt': 0, 'getsockopt': 0,
                  'getpeername': 0, 'write': 0, 'gettid': 0, 'prctl': 0, 'getrlimit': 0, 'clone': 0, 'set_robust_list': 0,
                  'recvmsg': 0, 'sendmsg': 0, 'writev': 0, 'lstat': 0, 'unlink': 0, 'stat': 0, 'pread': 0, 'fdatasync': 0,
                  'pwrite': 0, 'sched_yield': 0, 'rename': 0, 'mmap': 0, 'mprotect': 0, 'brk': 0, 'ftruncate': 0, 'select': 0,
                  'madvise': 0, 'fallocate': 0, 'shutdown': 0, 'exit': 0, 'procexit': 0, 'timerfd_settime': 0, 'rt_sigtimedwait': 0,
                  'wait4': 0, 'munmap': 0, 'recvfrom': 0, 'fcntl': 0, 'getpid': 0, 'access': 0, 'lseek': 0, 'fsync': 0, 'socket': 0,
                  'connect': 0, 'poll': 0, 'sendto': 0, 'bind': 0, 'times': 0, 'sysinfo': 0, 'setsid': 0, 'rt_sigprocmask': 0,
                  'execve': 0, 'arch_prctl': 0, 'rt_sigaction': 0, 'geteuid': 0, 'getppid': 0, 'dup': 0, 'kill': 0, 'exit_group': 0,
                  'signaldeliver': 0, 'sigreturn': 0, 'umask': 0, 'newfstatat': 0, 'unlinkat': 0, 'pipe': 0, 'vfork': 0, 'openat': 0,
                  'uname': 0, 'statfs': 0}


filter_flag = True


# Python code to merge dict using update() method
def merge(dict1, dict2):
    dict2.update(dict1)
    return dict2


def normalization(input_list):
    """Change number of occurrence input frequency distribution for a nested list"""
    output_list = []
    for fv in input_list:
        # change str to int
        fv = [int(c) for c in fv]
        fv_norm = [float(c)/sum(fv) for c in fv]
        output_list.append(fv_norm)
    return output_list


def strlist_2_list(strlist):
    """ reverse str(list) to a list; e.g '[1, 2, 4]' --> [1, 2, 4]"""
    x = ast.literal_eval(strlist)
    x = [n.strip() for n in x]
    return x


def common_element(a, b):
    """ compute if two lists have common memebers"""
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True
    else:
        return False


def generate_n_gram(n_gram, key_action, l=3):
    """ This function generates n-grams of length l"""

    if len(n_gram) < l:
        n_gram.append(key_action)
        return n_gram

    elif len(n_gram) == l:
        n_gram = n_gram[1:]
        n_gram.append(key_action)
        return n_gram

    else:
        raise ValueError


def n_gram_dict(filename, distinct_ngram_list=[]):
    """ This function extracts or update occurring distinct syscalls in the raw tracefile
    INPUT: filename --> new input"""
    n_gram = []
    with open(filename) as f:
        for line in f:
            if line.strip():  # skip empty line in the trace file
                line_list = line.split()
                try:
                    sys_call = line_list[6]
                except:
                    raise ValueError
                if sys_call == 'container':
                    continue

                n_gram = generate_n_gram(n_gram, sys_call, NGRAM_LENGTH)
                if len(n_gram) == NGRAM_LENGTH:
                    n_gram_s = str(n_gram)
                    if n_gram_s not in distinct_ngram_list:
                        distinct_ngram_list.append(n_gram_s)

    keys = distinct_ngram_list
    values = [0] * len(keys)
    distinct_ngram_dict = dict(zip(keys, values))

    return distinct_ngram_dict


def remove_ngram_fv(input_dict, remove_list=["futex", "sched_yield", "container"]):
    """ Remove keys contains elements in the remove list"""
    output_dict = {}
    for k in input_dict.keys():
        k = strlist_2_list(k)
        print(len(k))
        if len(k) != NGRAM_LENGTH:
            continue
        if not common_element(k, remove_list):
            output_dict[str(k)] = 0
    return output_dict


def fixed_length_feature_dict(filename, distinct_syscall_list=[]):
    """ This function extracts or update occurring distinct syscalls in the raw tracefile
    INPUT: filename --> new input
    distinct_syscall_list --> existing system call list"""
    with open(filename) as f:
        for line in f:
            if line.strip():  # skip empty line in the trace file
                line_list = line.split()
                try:
                    sys_call = line_list[6]
                except:
                    raise ValueError
                if sys_call not in distinct_syscall_list:
                    distinct_syscall_list.append(sys_call)
    keys = distinct_syscall_list
    values = [0] * len(keys)
    feature_dict = dict(zip(keys, values))
    if 'container' in feature_dict.keys():
        del feature_dict['container']
    return feature_dict


def parse_trace_ngram(filename, feature_dict, num_separate=10000):
    """This function changes traces into fix length n-gram feature vectors
    INPUT: filename --> raw tracefiles
    feature_dict: The fixed length feature vector"""
    feature_vector_list = []
    fv_index = 0
    # skip this trace if it is empty
    if os.stat(filename).st_size == 0:
        raise ValueError("The input file %s is empty!" % filename)
        return -1
    n_gram = []
    syscall_index = 0
    with open(filename) as f:
        for line in f:
            if line.strip():  # skip empty line in the trace file
                syscall_index += 1
                # print(syscall_index)
                line_list = line.split()
                try:
                    sys_call = line_list[6]
                except:
                    raise ValueError

                if filter_flag:
                    # if the action not belongs to one of the followings
                    if sys_call not in ["futex", "sched_yield", "container"]:
                    # if sys_call not in ["container"]:
                        n_gram = generate_n_gram(n_gram, sys_call, NGRAM_LENGTH)
                        if len(n_gram) == NGRAM_LENGTH:
                            n_gram_s = str(n_gram)
                            try:
                                feature_dict[n_gram_s] += 1
                            except KeyError:
                                print(key)
                                pass

                if syscall_index == num_separate:
                    print('seprate the tracefile')
                    syscall_index = 0
                    feature_vector = list(feature_dict.values())
                    #  skip if all the elements all zero
                    all_zero_flag = all([o == 0 for o in feature_vector])
                    if feature_vector not in feature_vector_list and not all_zero_flag:
                        fv_index += 1
                        feature_vector_list.append(feature_vector)

                    # reset dict values to zeros
                    feature_dict = dict.fromkeys(feature_dict, 0)

    return feature_vector_list


def parse_trace_tmp(filename, feature_dict, num_separate=10000, frequency_flag=True):
    """This function changes traces into feature vectors. The raw trace is separated with equal length num_separate
    OUTPUT: occurrence_dict: The no. of occurrences of features [syscall, ngram] in all traces of the database"""

    feature_vector_list = []
    # compute the occurrence of syscalls in the total database, currently only on the normal data; used for tf-idf
    occurrence_list = [0]*len(feature_dict.values())
    # total number of traces, for tf-idf
    N = 0

    # skip this trace if it is empty
    if os.stat(filename).st_size == 0:
        raise ValueError("The input file %s is empty!" % filename)
        return -1

    syscall_index = 0
    with open(filename) as f:
        for line in f:
            if line.strip():  # if the line is not empty
                syscall_index += 1
                line_list = line.split()
                try:
                    sys_call = line_list[6]
                except:
                    raise ValueError

                if filter_flag:
                    # if the action not belongs to one of the followings
                    if sys_call not in ["futex", "sched_yield", "container"]:
                        try:
                            feature_dict[sys_call] += 1
                        except:
                            print(sys_call)
                            pass

                if syscall_index == num_separate:
                    # print('seprate the tracefile')
                    N += 1
                    syscall_index = 0
                    feature_vector = list(feature_dict.values())

                    # validate if a specific syscall/ngram occurs in the feature vector, used to calculate idf (inverse
                    # document frequency)_
                    feature_vector_flag_tmp = [1 if c > 0 else 0 for c in feature_vector]
                    # element-wise addition of symbol occurrence number in the current trace and the historic data
                    occurrence_list = [sum(x) for x in zip(feature_vector_flag_tmp, occurrence_list)]
                    occurrence_dict = dict.fromkeys(feature_dict, 0)
                    # update the dict
                    occurrence_dict.update(zip(occurrence_dict, occurrence_list))

                    # compute the feature_vector list for each tracefile (separated with lenghth N), only remain those
                    # dictinct feature vectors
                    #  skip if all the elements all zero
                    all_zero_flag = all([o==0 for o in feature_vector])
                    if feature_vector not in feature_vector_list and not all_zero_flag:
                        feature_vector_list.append(feature_vector)

                    # reset dict values to zeros
                    feature_dict = dict.fromkeys(feature_dict, 0)

    return feature_vector_list, occurrence_dict, N


def df_idf(feature_vector_list, occurrence_dict, N):
    """ This function generates nested feature vectors weighted tf-idf"""
    dfidf_list = []
    # compute the idfparameter
    p_list = []
    fv_idf_list = []
    for v in occurrence_dict.values():
        v = float(v)
        p = math.log(N/(v+1), 10)
        p_list.append(p)

    for fv in feature_vector_list:
        fv_idf = [a*b for a, b in zip(p_list, fv)]
        fv_idf_list.append(fv_idf)
    return fv_idf_list


def feature_vector_csv_generator(feature_vector_list, csv_filename):
    """Write the feature vector list into csv file"""

    with open(csv_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(feature_vector_list)


def generate_csv_file(rawtrace_file, feature_dict_file, csv_filename, Flag, Read_ngram_dict=True):
    """ parse raw trace and generate csv file serving as input of classifiers. Flag = 0 for tf, 1 for idf-tf,
    2 for n-gram"""

    feature_dict = json.load(open(feature_dict_file))

    if Flag == 0:
        feature_vector_list, occurrence_dict, N = parse_trace_tmp(rawtrace_file, feature_dict)
        print(feature_vector_list)
        feature_vector_list = normalization(feature_vector_list)
        feature_vector_csv_generator(feature_vector_list, csv_filename)

    if Flag == 1:
        feature_vector_list, occurrence_dict, N = parse_trace_tmp(rawtrace_file, feature_dict)
        feature_vector_list = normalization(feature_vector_list)
        fv_idf_list = df_idf(feature_vector_list, occurrence_dict, N)
        feature_vector_csv_generator(fv_idf_list, csv_filename)

    if Flag == 2:
        if Read_ngram_dict:
            # Read data from file:
            feature_dict = json.load(open("MONGO_FEATURE_DICT_NGRAM.json"))

        else: # no FEATURE_DICT FOR N-GRAM EXISTS
            feature_dict = n_gram_dict(rawtrace_file)
            # Serialize data into file:
            json.dump(feature_dict, open("MONGO_FEATURE_DICT_NGRAM.json", 'w'))

        feature_vector_list = parse_trace_ngram(rawtrace_file, feature_dict)
        feature_vector_csv_generator(feature_vector_list, csv_filename)


def generate_csv_file_loop(rawtrace_file_list, feature_dict_file='FEATURE_DICT.json'):
    for rawtrace in rawtrace_file_list:
        rawtrace_file = rawtrace[0]
        csv_filename = rawtrace[1]
        flag_list = [0, 1]
        for flag in flag_list:
            if flag == 0:
                csv_filename_full = csv_filename + "_tf.csv"
            elif flag == 1:
                csv_filename_full = csv_filename + "_tfidf.csv"
            print(csv_filename_full)
            generate_csv_file(rawtrace_file, feature_dict_file, csv_filename_full, flag)


def main():
    # rawtrace_file = 'ML_algorithm/co7_ubuntu/co7_ubuntu_db0_0'
    # generate_csv_file(rawtrace_file, 'FEATURE_DICT.json', 'mongodb/mb_normal', 1)
    rawtrace_file_list = [('ML_algorithm/co1_ubuntu_db0/tracefile-0', 'ML_algorithm/ml_1_normal'),
                          ('ML_algorithm/co2_ubuntu_db0/tracefile-0', 'ML_algorithm/ml_2_normal'),
                          ('ML_algorithm/co3_ubuntu_db0/tracefile-0', 'ML_algorithm/ml_3_normal'),
                          ('ML_algorithm/co4_ubuntu_db0/tracefile-0', 'ML_algorithm/ml_4_normal'),
                          ('ML_algorithm/co7_ubuntu_db0/tracefile-0', 'ML_algorithm/ml_7_normal'),
                          ]
    # generate_csv_file_loop(rawtrace_file_list)

if __name__ == "__main__":
    main()
