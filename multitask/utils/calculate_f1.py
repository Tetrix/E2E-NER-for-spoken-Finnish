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
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    # calculate recall
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    
    precision = np.array(precision)
    recall = np.array(recall)

    # calculate f1
    if precision + recall == 0:
        f1 = 0.0
    else:
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
    print('per precision: %.4f    per recall: %.4f    per f1: %.4f' %(per_precision.mean(), per_recall.mean(), per_f1))

    loc_precision, loc_recall, loc_f1 = calculate_individual_score(all_predicted, all_true, 'LOC')
    print('loc precision: %.4f    loc recall: %.4f    loc f1: %.4f' %(loc_precision.mean(), loc_recall.mean(), loc_f1))

    org_precision, org_recall, org_f1 = calculate_individual_score(all_predicted, all_true, 'ORG')
    print('org precision: %.4f    org recall: %.4f    org f1: %.4f' %(org_precision.mean(), org_recall.mean(), org_f1))


    per_tp, per_fp, per_fn = calculate_average_score(all_predicted, all_true, 'PER')
    loc_tp, loc_fp, loc_fn = calculate_average_score(all_predicted, all_true, 'LOC')
    org_tp, org_fp, org_fn = calculate_average_score(all_predicted, all_true, 'ORG')


    if (per_tp + loc_tp + per_fp + loc_fp) == 0:
        micro_avg_precision = 0
    else:
        micro_avg_precision = (per_tp + loc_tp) / (per_tp + loc_tp + per_fp + loc_fp)

    if (per_tp + loc_tp + per_fn + loc_fn) == 0:
        micro_avg_recall = 0
    else:
        micro_avg_recall = (per_tp + loc_tp) / (per_tp + loc_tp + per_fn + loc_fn)

    if (micro_avg_precision + micro_avg_recall) == 0:
        avg_f1 = 0
    else:
        avg_f1 = 2 * (micro_avg_precision * micro_avg_recall) / (micro_avg_precision + micro_avg_recall)

    print('micro avg precision: %.4f    micro avg recall: %.4f    avg f1: %.4f' %(micro_avg_precision, micro_avg_recall, avg_f1))


