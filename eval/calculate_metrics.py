import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.special import rel_entr

def strint(x):
    return str(int(x))

def generate_cond_cont(dataset, mr, size, sample_id, attn_var, y_loc):
    complete_data_path = '../training_data/samples/' + dataset + '/complete_' + str(mr) + '_' + str(size) + '/sample_' + str(sample_id) + '.csv'
    data = pd.read_csv(complete_data_path, header=None)

    # get all possible levels' combination for categorical variable that we specified
    all_levels = [np.unique(data.iloc[:,i]).tolist() for i in attn_var]
    all_levels_comb = list(product(*all_levels))

    # extract conditional distributions for complete data
    cond_dist_complete = dict.fromkeys(all_levels_comb, None)
    condtag = []
    
    for index, item in data.iterrows():
        cond = '-'.join([strint(item[i]) for i in attn_var])
        condtag.append(cond)
    
    data['cond'] = condtag
    
    for key in all_levels_comb:
        str_key = '-'.join([strint(key[i]) for i in range(len(key))])
        cond_dist_complete[key] = data.loc[data['cond'] == str_key][y_loc].values.tolist()
    
    pdata = data.copy()
    pdata[y_loc] = np.random.permutation(pdata[y_loc])

    return all_levels, all_levels_comb, cond_dist_complete, pdata, condtag

# output MSE for each pair of specified conditional distribution in one sample
# extract conditional dataset from imputed datasets
def metric_comparison(method, imputed_data_folder, quant_list,
                      all_levels, all_levels_comb, cond_dist_complete, attn_var, condtag,
                      sample_id, impute_num):

    cond_dist_imputed = {key:[] for key in all_levels_comb}
    mse_list = {key:[] for key in all_levels_comb}
    mae_list = {key:[] for key in all_levels_comb}

    for i in range(impute_num):
        current_imputed_dir = imputed_data_folder + 'imputed_' + str(sample_id) + '_' + str(i) + '.csv'
        imputed_data = pd.read_csv(current_imputed_dir, header=None)        
        imputed_data['cond'] = condtag

        for key in all_levels_comb:
            str_key = '-'.join([strint(key[i]) for i in range(len(key))])
            cond_imputed_data = imputed_data.loc[imputed_data['cond'] == str_key][5].values.tolist()
            
            # add data to imputed conditional distribution
            cond_dist_imputed[key] = cond_dist_imputed[key] + cond_imputed_data

            # calculate metrics
            y_true = cond_dist_complete[key]
            mse_list[key].append(mean_squared_error(y_true, cond_imputed_data))
            mae_list[key].append(mean_absolute_error(y_true, cond_imputed_data))

    # calculate average
    mse_quantile = dict.fromkeys(all_levels_comb, None)
    mae_quantile = dict.fromkeys(all_levels_comb, None)
    for key in all_levels_comb:
        mse_quantile[key] = np.quantile(mse_list[key], quant_list)
        mae_quantile[key] = np.quantile(mae_list[key], quant_list)
        
    return cond_dist_imputed, mse_quantile, mae_quantile


# output estimated KL for each pair of specified conditional distribution in one sample
# extract conditional dataset from imputed datasets
def kl_comparison(imputed_data_folder, bin_list,
                      all_levels_comb, cond_dist_complete, condtag,
                      sample_id, impute_num, y_loc):

    cond_dist_imputed = {key:[] for key in all_levels_comb}
    kl_val = {key:[] for key in all_levels_comb}

    for i in range(impute_num):
        current_imputed_dir = imputed_data_folder + 'imputed_' + str(sample_id) + '_' + str(i) + '.csv'
        imputed_data = pd.read_csv(current_imputed_dir, header=None)        
        imputed_data['cond'] = condtag

        for key in all_levels_comb:
            str_key = '-'.join([strint(key[i]) for i in range(len(key))])
            cond_imputed_data = [x for x in imputed_data.loc[imputed_data['cond'] == str_key][y_loc].values.tolist()]
            
            # add data to imputed conditional distribution
            cond_dist_imputed[key] = cond_dist_imputed[key] + cond_imputed_data

    def bin_continuous(x):
        neg_x = [item * (-1) for item in x]
        bin_counts = [len([item for item in neg_x if item == -1])]
        bin_counts.extend(pd.cut(neg_x, bins = bin_list).value_counts().tolist())
        bin_counts = [item / len(x) for item in bin_counts]
        return bin_counts
    
    for key in all_levels_comb:
        # calculate metrics
        y_true_bin = bin_continuous(cond_dist_complete[key])
        y_pred_bin = bin_continuous(cond_dist_imputed[key])
        kl_val[key].append(round(sum(rel_entr(y_true_bin, y_pred_bin)),6))
        
    return cond_dist_imputed, kl_val