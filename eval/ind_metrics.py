import numpy as np
import pandas as pd
from sklearn import metrics as mtr
from scipy.spatial.distance import jensenshannon

def get_prob(labels, max_label):
    test = pd.Series([i for i in labels])
    counts = []
    for i in range(max_label):
        counts.append(sum(x == i for x in labels))
    counts = [x / len(test) for x in counts]
    return counts

def categorical_metrics(pred, target, compare_loc):    
    acc = mtr.accuracy_score(target[compare_loc], pred[compare_loc])
    precision, recall, fscore, _ = mtr.precision_recall_fscore_support(target[compare_loc], pred[compare_loc], average='weighted', zero_division=np.nan)

    max_label = int(max(target))
    p_pred = get_prob(pred, max_label)
    p_target = get_prob(target, max_label)
    js_div = jensenshannon(p_pred, p_target)
    return acc, precision, recall, fscore, js_div

def numeric_metrics(pred, target, compare_loc):
    norm_pred = pred / sum(pred)
    norm_pred = norm_pred.astype(np.float64)
    norm_target = target / sum(target)
    norm_target = norm_target.astype(np.float64)

    mse = mtr.mean_squared_error(norm_target[compare_loc], norm_pred[compare_loc])
    mae = mtr.mean_absolute_error(norm_target[compare_loc], norm_pred[compare_loc])
    js_div = jensenshannon(norm_pred, norm_target)
    return mse, mae, js_div

def get_metrics(complete_data, mask_data, imputed_data_folder, data_level, impute_num, sample_id =0):
    cat_index = data_level['cat_index']
    num_index = data_level['num_index']
    
    js_div_list = np.zeros(shape=(complete_data.shape[1], impute_num))
    acc_list = np.zeros(shape=(complete_data.shape[1], impute_num))
    precision_list = np.zeros(shape=(complete_data.shape[1], impute_num))
    recall_list = np.zeros(shape=(complete_data.shape[1], impute_num))
    fscore_list = np.zeros(shape=(complete_data.shape[1], impute_num))
    mse_list = np.zeros(shape=(complete_data.shape[1], impute_num))
    mae_list = np.zeros(shape=(complete_data.shape[1], impute_num))

    for i in range(impute_num):
        current_imputed_dir = imputed_data_folder + 'imputed_' + str(sample_id) + '_' + str(i) + '.csv'
        imputed_data = np.loadtxt(current_imputed_dir, delimiter=",").astype(np.float32)

        for y_loc in cat_index:
            compare_loc = np.where(mask_data[:,y_loc])[0]
            if len(compare_loc) == 0:
                continue
            target = complete_data[:, y_loc]
            pred = imputed_data[:, y_loc]
            acc, precision, recall, fscore, js_div = categorical_metrics(pred, target, compare_loc)
            
            js_div_list[y_loc, i] = js_div
            acc_list[y_loc, i] = acc
            precision_list[y_loc, i] = precision
            recall_list[y_loc, i] = recall
            fscore_list[y_loc, i] = fscore
        
        for y_loc in num_index:
            compare_loc = np.where(mask_data[:,y_loc])[0]
            if len(compare_loc) == 0:
                continue
            target = complete_data[:, y_loc]
            pred = imputed_data[:, y_loc]
            mse, mae, js_div = numeric_metrics(pred, target, compare_loc)
            
            mse_list[y_loc, i] = mse
            mae_list[y_loc, i] = mae
            js_div_list[y_loc, i] = js_div
        
        metrics = {'acc':acc_list, 'precision':precision_list, 'recall': recall_list, 'fscore':fscore_list,
                   'mse': mse_list, 'mae': mae_list, 'jsd': js_div_list}
    
    return metrics
