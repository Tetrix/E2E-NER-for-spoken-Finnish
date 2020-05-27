import numpy as np



def calculate_individual_score(all_predicted, all_true, entity_tag):
    precision = []
    recall = []
    tp = 0
    fp = 0
    fn = 0
    for seq in range(len(all_true)):
        for tag in range(len(all_true[seq])):
            if all_predicted[seq][tag] == entity_tag and all_true[seq][tag] == entity_tag:
                tp += 1
            elif all_predicted[seq][tag] == entity_tag and all_true[seq][tag] != entity_tag:
                fp += 1
            elif all_predicted[seq][tag] != entity_tag and all_true[seq][tag] == entity_tag:
                fn += 1

    # calculate precision
    try:
        precision = tp / (tp + fp)
    except:
        precision = 0

    # calculate recall
    try:
        recall = tp / (tp + fn)
    except:
        recall = 0
    
    precision = np.array(precision)
    recall = np.array(recall)

    # calculate f1
    try:
        f1 = 0.0
    except:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def calculate_average_score(all_predicted, all_true, entity_tag):
    tp = 0
    fp = 0
    fn = 0
    for seq in range(len(all_true)):      
        for tag in range(len(all_true[seq])):
            if all_predicted[seq][tag] == entity_tag and all_true[seq][tag] == entity_tag:
                tp += 1
            elif all_predicted[seq][tag] == entity_tag and all_true[seq][tag] != entity_tag:
                fp += 1
            elif all_predicted[seq][tag] != entity_tag and all_true[seq][tag] == entity_tag:
                fn += 1
    
    return tp, fp, fn


def print_scores(all_predicted, all_true):
    per_precision, per_recall, per_f1 = calculate_individual_score(all_predicted, all_true, 'PER')
    print('per precision: %.4f    per recall: %.4f    per_f1: %.4f' %(per_precision.mean(), per_recall.mean(), per_f1))

    loc_precision, loc_recall, loc_f1 = calculate_individual_score(all_predicted, all_true, 'LOC')
    print('loc precision: %.4f    loc recall: %.4f    loc_f1: %.4f' %(loc_precision.mean(), loc_recall.mean(), loc_f1))


    per_tp, per_fp, per_fn = calculate_average_score(all_predicted, all_true, 'PER')
    loc_tp, loc_fp, loc_fn = calculate_average_score(all_predicted, all_true, 'LOC')
    
    try:
        micro_avg_precision = (per_tp + loc_tp) / (per_tp + loc_tp + per_fp + loc_fp)
    except:
        micro_avg_precision = 0.0
    try:
        micro_avg_recall = (per_tp + loc_tp) / (per_tp + loc_tp + per_fn + loc_fn)
    except:
        micro_avg_recall = 0.0
    try:
        avg_f1 = 2 * (micro_avg_precision * micro_avg_recall) / (micro_avg_precision + micro_avg_recall)
    except:
        avg_f1 = 0.0

    print('micro avg precision: %.4f    micro avg recall: %.4f    avg f1: %.4f' %(micro_avg_precision, micro_avg_recall, avg_f1))


