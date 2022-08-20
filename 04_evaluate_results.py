import pandas as pd
import numpy as np
import json
import os
from os.path import join, isfile


multi_result_cols = [
    "TrainSet", "TrainExperiment", "TestSet", "TestExperiment", "ModelName", "Splits",
    "Jaccard_beam_1", "Jaccard_beam_3", "Jaccard_beam_5", "Jaccard_beam_10", "Jaccard_beam_20",
    "Recall_beam_1", "Recall_beam_3", "Recall_beam_5", "Recall_beam_10", "Recall_beam_20"]

single_result_cols = [
    "TrainSet", "TrainExperiment", "TestSet", "TestExperiment", "ModelName", "Splits",
    "Top_1_Acc", "Top_2_Acc", "Top_3_Acc", "Top_4_Acc", "Top_5_Acc", 
    "Top_1_Delta", "Top_2_Delta", "Top_3_Delta", "Top_4_Delta", "Top_5_Delta"]

save_path = "outputs/"


def create_pickle_path_dicts(results_dict_path, splice_type):
    retVal = {}
    onlyfiles = [f for f in os.listdir(results_dict_path) if isfile(join(results_dict_path, f))]
    filtered_files = [f for f in onlyfiles if (splice_type in f) and (f.endswith(".json"))]
    
    retVal = {}
    for f in filtered_files:
        with open(results_dict_path+"/"+f, "r") as fp:
                retVal[f] = json.load(fp)
    return retVal


def check_position_accuracy_delta(result_dict, top_k, is_transformer=True):
    if is_transformer:
        r_dict = result_dict["beam_5"]
    else:
        r_dict = result_dict
        for k in r_dict.keys():
            p_temp = r_dict[k]["predictions"]["0"]
            r_temp = r_dict[k]["real"]
            for i in range(5):
                r_dict[k]["predictions"][str(i)] = [0, p_temp[i]]
            r_dict[k]["real"] = [0, r_temp[0], 2]

    
    
    keys = list(r_dict.keys())
    
    

    retVal_delta = {1: {},2: {},3: {},4: {},5: {}}
    retVal_acc = {1: {},2: {},3: {},4: {},5: {}}
    total_position_frequency = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0
    }
    pos_fix = 1

    for k in keys: # Samples
        for pos in range(pos_fix,len(r_dict[k]["real"])-pos_fix):

            real_value = r_dict[k]["real"][pos]
            total_position_frequency[pos+ (1 - pos_fix)] += 1
            for i in range(1,top_k+1): # Top K
                preds_col = []
                for ii in range(i):
                    pred_value = r_dict[k]["predictions"][str(ii)][pos]
                    preds_col.append(pred_value)
                
                preds_col = np.asarray(preds_col)
                if real_value in preds_col:
                    if i in retVal_acc[pos+ (1 - pos_fix)].keys():
                        retVal_acc[pos+ (1 - pos_fix)][i] += 1
                    else:
                        retVal_acc[pos+ (1 - pos_fix)][i] = 1


                preds_col = np.abs(preds_col - real_value)
                min_delta = np.min(preds_col)

                if i in retVal_delta[pos+ (1 - pos_fix)].keys():
                    retVal_delta[pos+ (1 - pos_fix)][i].append(min_delta)
                else:
                    retVal_delta[pos+ (1 - pos_fix)][i] = [min_delta]

    for splice in retVal_delta.keys():
        for k in retVal_delta[splice].keys():
            retVal_delta[splice][k] = np.mean(retVal_delta[splice][k])

    for k in retVal_acc.keys():
        freq = total_position_frequency[k]
        for t_k in retVal_acc[k].keys():
            retVal_acc[k][t_k] /= freq

    return retVal_acc, retVal_delta

def calc_jaccard(result_dict):
    beam_keys = list(result_dict.keys())
    retVal = {}
    
    for beam in beam_keys:
        hit_list = []
        treffer_total = 0

        keys = list(result_dict[beam].keys())
        for k in keys:
            pred_set = np.asarray(result_dict[beam][k]["predictions"]["0"]) 
            pred_set = np.unique(pred_set)
            r_set = np.asarray(result_dict[beam][k]["real"])

            
            r_set = r_set[(r_set != 0) & (r_set != 2)]
            pred_set = pred_set[(pred_set != 0) & (pred_set != 2)]

            # window size /2 = 1 sec /5 = 2.5 sec /10 = 5 sec
            r_set = np.ceil(r_set/1)
            pred_set = np.ceil(pred_set/1)

            total_set = np.concatenate([r_set, pred_set])
            total_set = np.unique(total_set)
            pred_set = np.unique(pred_set)
            r_set = np.unique(r_set)

            a_cat_b = np.concatenate([r_set, pred_set])
            a_cat_b_unique ,counts = np.unique(a_cat_b, return_counts=True)
            intersection = a_cat_b_unique[np.where(counts>1)]

            hits = len(intersection)
            treffer_total += hits
            hits = hits/len(total_set)
            hit_list.append(hits)
        retVal[beam] = np.mean(hit_list)
    return retVal

def calc_recall(result_dict):
    beam_keys = list(result_dict.keys())
    retVal = {}
    
    for beam in beam_keys:
        hit_list = []
        treffer_total = 0

        keys = list(result_dict[beam].keys())
        for k in keys:
            pred_set = np.asarray(result_dict[beam][k]["predictions"]["0"])
            pred_set = np.unique(pred_set)
            r_set = np.asarray(result_dict[beam][k]["real"])

            
            r_set = r_set[(r_set != 0) & (r_set != 2)]
            pred_set = pred_set[(pred_set != 0) & (pred_set != 2)]

            # window size /2 = 1 sec /5 = 2.5 sec /10 = 5 sec
            r_set = np.ceil(r_set/1)
            pred_set = np.ceil(pred_set/1)

            pred_set = np.unique(pred_set)
            r_set = np.unique(r_set)

            a_cat_b = np.concatenate([r_set, pred_set])
            a_cat_b_unique ,counts = np.unique(a_cat_b, return_counts=True)
            intersection = a_cat_b_unique[np.where(counts>1)]

            hits = len(intersection)
            treffer_total += hits
            hits = hits/len(r_set)
            hit_list.append(hits)
        retVal[beam] = np.mean(hit_list)
    return retVal

def calc_recall_greedy(result_dict):
    retVal = {}
    keys = list(result_dict.keys())
    
    for topk in [1,2,3,4,5]:
        hit_list = []
        for k in keys:
            r_set = np.asarray(result_dict[k]["real"])
            pred_set = []
            for tk in range(topk):
                for pos in ["0", "1", "2", "3", "4"]:
                    pred_set.append(result_dict[k]["predictions"][pos][tk])
                
            pred_set = np.asarray(pred_set)
            r_set = r_set[(r_set != 0) & (r_set != 2)]
            pred_set = pred_set[(pred_set != 0) & (pred_set != 2)]

            # window size /2 = 1 sec /5 = 2.5 sec /10 = 5 sec
            #r_set = np.ceil(r_set/1)
            #pred_set = np.ceil(pred_set/1)

            pred_set = np.unique(pred_set)
            r_set = np.unique(r_set)

            a_cat_b = np.concatenate([r_set, pred_set])
            a_cat_b_unique ,counts = np.unique(a_cat_b, return_counts=True)
            intersection = a_cat_b_unique[np.where(counts>1)]

            hits = len(intersection)
            hits = hits/len(r_set)
            hit_list.append(hits)
        retVal["top"+str(topk)] = np.mean(hit_list)
    return retVal

def split_name_in_parts(name):
    parts = name.split("-")
    train_set = parts[0]
    train_experiment = parts[1]
    test_set = parts[2]
    test_experiment = parts[3]
    model_name = parts[4]
    splits = parts[5][:-5]
    return train_set, train_experiment, test_set, test_experiment, model_name, splits
    
def prepare_dict(result_dictionary, beam):
    retDict = {beam:{}}
    for k in result_dictionary.keys():
        top1_preds = []
        for kk in result_dictionary[k]["predictions"]:
            top1_preds.append(result_dictionary[k]["predictions"][kk][0])
        real = result_dictionary[k]["real"]

        real = [value for value in real if value != 3]
        top1_preds = [value for value in top1_preds if value != 3]

        if len(real) == 0:
            real.append(3)
        if len(top1_preds) == 0:
            top1_preds.append(3)


        retDict[beam][k] = {
            "predictions": {
                "0": top1_preds
            },
            "probabilities": {
                "0": 1.0
            },
            "real": real
        }
    return retDict

def main():
    results_path = "models/transformer/ace"
    splice_type = "0_1" #0_1, 3_3, 0_5

    result_files = create_pickle_path_dicts(results_path, splice_type)
    keys = list(result_files.keys())

    result_df = []
    for k in keys:
        print(k)
        train_set, train_experiment, test_set, test_experiment, model_name, splits = split_name_in_parts(k)
        if splice_type == "0_1":
            is_transformer = True
            if (model_name != "transformer") and (model_name != "transformer_m"):
                    is_transformer = False
            acc_dict, delta_dict = check_position_accuracy_delta(result_files[k], top_k=5, is_transformer=is_transformer)
            result_df.append([
                train_set, train_experiment, test_set, test_experiment, model_name, splits,
                acc_dict[1][1], acc_dict[1][2], acc_dict[1][3], acc_dict[1][4], acc_dict[1][5], 
                delta_dict[1][1], delta_dict[1][2], delta_dict[1][3], delta_dict[1][4], delta_dict[1][5]
            ])
        else:
            try:
                if (model_name != "transformer") and (model_name != "transformer_m"):
                    result_files[k] = prepare_dict(result_files[k], "beam_1")

                jaccard_dict = calc_jaccard(result_files[k])
                recall_dict = calc_recall(result_files[k])
                result_df.append([
                    train_set, train_experiment, test_set, test_experiment, model_name, splits,
                    jaccard_dict["beam_1"] if "beam_1" in jaccard_dict else 0, 
                    jaccard_dict["beam_3"] if "beam_3" in jaccard_dict else 0,
                    jaccard_dict["beam_5"] if "beam_5" in jaccard_dict else 0,
                    jaccard_dict["beam_10"] if "beam_10" in jaccard_dict else 0,
                    jaccard_dict["beam_20"] if "beam_20" in jaccard_dict else 0,
                    recall_dict["beam_1"] if "beam_1" in recall_dict else 0, 
                    recall_dict["beam_3"] if "beam_3" in recall_dict else 0,
                    recall_dict["beam_5"] if "beam_5" in recall_dict else 0,
                    recall_dict["beam_10"] if "beam_10" in recall_dict else 0,
                    recall_dict["beam_20"] if "beam_20" in recall_dict else 0,
                ])
                
            except:
                print("Failed: ",k)

    if splice_type == "0_1":
        result_df = pd.DataFrame(result_df, columns=single_result_cols)
        result_df.to_csv(save_path+splice_type+"_evaluation.csv", index=False, sep=";")
    else:
        result_df = pd.DataFrame(result_df, columns=multi_result_cols)
        result_df.to_csv(save_path+splice_type+"_evaluation.csv", index=False, sep=";")
    

if __name__ == "__main__":
    main()