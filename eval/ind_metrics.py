import numpy as np
import pandas as pd
from sklearn import metrics as mtr
from scipy.spatial.distance import jensenshannon

def categorical_metrics(pred, target):
    norm_pred = pred / sum(pred)
    norm_pred = norm_pred.astype(np.float64)
    norm_target = target / sum(target)
    norm_target = norm_target.astype(np.float64)
    
    acc = mtr.accuracy_score(target, pred)
    precision, recall, fscore, _ = mtr.precision_recall_fscore_support(target, pred, average='weighted')
    js_div = jensenshannon(norm_pred, norm_target)
    return acc, precision, recall, fscore, js_div

def numeric_metrics(pred, target):
    norm_pred = pred / sum(pred)
    norm_pred = norm_pred.astype(np.float64)
    norm_target = target / sum(target)
    norm_target = norm_target.astype(np.float64)

    mse = mtr.mean_squared_error(target, pred)
    mae = mtr.mean_absolute_error(target, pred)
    js_div = jensenshannon(norm_pred, norm_target)
    return mse, mae, js_div

def get_metrics(complete_data, imputed_data_folder, data_level, impute_num, sample_id =0):
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
            target = complete_data[:, y_loc]
            pred = imputed_data[:, y_loc]
            acc, precision, recall, fscore, js_div = categorical_metrics(pred, target)
            
            js_div_list[y_loc, i] = js_div
            acc_list[y_loc, i] = acc
            precision_list[y_loc, i] = precision
            recall_list[y_loc, i] = recall
            fscore_list[y_loc, i] = fscore
        
        for y_loc in num_index:
            target = complete_data[:, y_loc]
            pred = imputed_data[:, y_loc]
            mse, mae, js_div = numeric_metrics(pred, target)
            
            mse_list[y_loc, i] = mse
            mae_list[y_loc, i] = mae
            js_div_list[y_loc, i] = js_div
        
        metrics = {'acc':acc_list, 'precision':precision_list, 'recall': recall_list, 'fscore':fscore_list,
                   'mse': mse_list, 'mae': mae_list, 'jsd': js_div_list}
    
    return metrics
