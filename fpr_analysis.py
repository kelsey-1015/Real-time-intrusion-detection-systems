import json
import plot

def check_standalone(numlist, interval = 5 ):
    """Check the number of standalone sequence"""
    standalone_flag_list = []
    total_fp = len(numlist)
    for i in range(len(numlist)):
        if i == 0:
            num_current = numlist[i]
            num_next = numlist[i + 1]
            if num_next - num_current != interval:
                standalone_flag = True
            else:
                standalone_flag = False

        if i == len(numlist) -1:
            num_current = numlist[i]
            num_before = numlist[i -1]
            if num_current - num_before != interval:
                standalone_flag = True
            else:
                standalone_flag = False
        else:
            num_before = numlist[i - 1]
            num_current = numlist[i]
            num_next = numlist[i + 1]
            if num_next - num_current != interval and num_current - num_before != interval:
                standalone_flag = True
            else:
                standalone_flag = False

        standalone_flag_list.append(standalone_flag)
        num_standalone = sum(standalone_flag_list)

    num_rm_standalone = total_fp - num_standalone

    return num_rm_standalone


def parse_jsonfile(json_file, algorithm_name, segment_length, test_normal_sample_num):
    dict = json.load(open(json_file))
    key_nu = "0.01"
    FPR_original = dict[algorithm_name][segment_length][key_nu][0]
    FP_ss_list = dict[algorithm_name][segment_length][key_nu][-1]
    num_FP_rm = check_standalone(FP_ss_list)
    FPR_new = num_FP_rm/test_normal_sample_num
    return FPR_original, FPR_new


def parse_loop_sl(json_file, alglorithm_name, segment_length_list, SL_TN_number_mongodb):

    FPR_original_list = []
    FPR_new_list = []
    for sl in segment_length_list:
        test_normal_sample_num = SL_TN_number_mongodb[int(sl)]
        sl = str(sl)
        FPR_original, FPR_new = parse_jsonfile(json_file, alglorithm_name, sl, test_normal_sample_num)
        FPR_original_list.append(FPR_original)
        FPR_new_list.append(FPR_new)

    return FPR_original_list, FPR_new_list


def main():

    segment_length_list = [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 50000]
    SL_TN_number_mongodb = {1000: 7460, 2000: 3730, 5000: 1490, 10000: 750, 15000: 500, 20000: 370, 25000: 300,
                            30000: 250, 50000: 150}

    json_file_tf = 'fpr_results/mongodb_fpr_ss_tf.json'
    json_file_ngram = 'fpr_results/mongodb_fpr_ss_NGRAM.json'
    alglorithm_name_tf = "linear_TF"
    alglorithm_name_ngram = "linear_N_GRAM"

    FPR_original_list_tf, FPR_new_list_tf = parse_loop_sl(json_file_tf, alglorithm_name_tf, segment_length_list,
                                                    SL_TN_number_mongodb)
    FPR_original_list_ngram, FPR_new_list_ngram = parse_loop_sl(json_file_ngram, alglorithm_name_ngram, segment_length_list,
                                                    SL_TN_number_mongodb)
    plot.plot_fpr_reduction(FPR_original_list_tf, FPR_new_list_tf, FPR_original_list_ngram, FPR_new_list_ngram,
                            segment_length_list)


if __name__ == "__main__":

    main()