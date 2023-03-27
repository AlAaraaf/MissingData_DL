from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import argparse

from utils.utils import table_to_latex

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type = str, required=True)
    parser.add_argument("-output", type = str, required=True)
    return parser.parse_args()

args = parse_args()
# Load data
save_name = args.dataset
numeric_variable_nums = dict([('boston', 12), ('house',8),('sim_1', 0),('sim_2',0), ('sim_1_tiny',0), ('sim_2_tiny',0)])
if save_name not in numeric_variable_nums.keys() and save_name != 'income':
    sys.exit("Wrong Dataset!")

miss_mechanism = "MCAR"
file_name = 'data/' + save_name + '.csv'
data_df = pd.read_csv(file_name)
data_x = data_df.values.astype(np.float32)
save_path = "../metrics/{}/{}".format(save_name, miss_mechanism)
save_mode = "w"
save_loc = args.output + "_relative_metrics50.tex"

# check numerical and categorical variable numbers.
if save_name == 'income':
    num_index = list([9, 16, 18, 19])
    cat_index = list(set(range(data_df.shape[1])).difference(set(num_index)))
else:
    num_index = numeric_variable_nums[save_name]
    cat_index = data_df.shape[1]-numeric_variable_nums[save_name]

# load cat
if cat_index > 0:
    mar_relmse = pd.DataFrame(np.load(os.path.join(save_path, "mar_rel_mse.npy"), allow_pickle=True).item())
    mar_relmse_table = mar_relmse.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    biv_relmse = pd.DataFrame(np.load(os.path.join(save_path, "biv_rel_mse.npy"), allow_pickle=True).item())
    biv_relmse_table = biv_relmse.quantile([0.1, 0.25, 0.5, 0.75, 0.9])

    mar_nbias = pd.DataFrame(np.load(os.path.join(save_path, "mar_nbias.npy"), allow_pickle=True).item())
    mar_nbias_table = mar_nbias.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    mar_nvar = pd.DataFrame(np.load(os.path.join(save_path, "mar_nvar.npy"), allow_pickle=True).item())
    mar_nvar_table = mar_nvar.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    biv_nbias = pd.DataFrame(np.load(os.path.join(save_path, "biv_nbias.npy"), allow_pickle=True).item())
    biv_nbias_table = biv_nbias.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    biv_nvar = pd.DataFrame(np.load(os.path.join(save_path, "biv_nvar.npy"), allow_pickle=True).item())
    biv_nvar_table = biv_nvar.quantile([0.1, 0.25, 0.5, 0.75, 0.9])

    mar_relbias = pd.DataFrame(np.load(os.path.join(save_path, "mar_relbias.npy"), allow_pickle=True).item())
    mar_relbias_table = mar_relbias.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    mar_relvar = pd.DataFrame(np.load(os.path.join(save_path, "mar_relvar.npy"), allow_pickle=True).item())
    mar_relvar_table = mar_relvar.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    biv_relbias = pd.DataFrame(np.load(os.path.join(save_path, "biv_relbias.npy"), allow_pickle=True).item())
    biv_relbias_table = biv_relbias.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    biv_relvar = pd.DataFrame(np.load(os.path.join(save_path, "biv_relvar.npy"), allow_pickle=True).item())
    biv_relvar_table = biv_relvar.quantile([0.1, 0.25, 0.5, 0.75, 0.9])

    # generate latex for cat
    table_to_latex(mar_relmse_table, biv_relmse_table, "relative mean squared error", "categorical", float_format = "%.2f", save_mode=save_mode, save_loc=save_loc)
    save_mode = "a"
    # table_to_latex(mar_nbias_table, biv_nbias_table, "normalized bias", "categorical", float_format = "%.2f", save_mode=save_mode, save_loc=save_loc, percentage=True)
    # table_to_latex(mar_nvar_table, biv_nvar_table, "normalized variance", "categorical", float_format = "%.2f", save_mode=save_mode, save_loc=save_loc)
    # table_to_latex(mar_relbias_table, biv_relbias_table, "relative bias", "categorical", float_format = "%.2f", save_mode=save_mode, save_loc=save_loc)
    # table_to_latex(mar_relvar_table, biv_relvar_table, "relative variance", "categorical", float_format = "%.2f", save_mode=save_mode, save_loc=save_loc)


# load cont
if num_index > 0:
    mar_bin_relmse = pd.DataFrame(np.load(os.path.join(save_path, "mar_bin_rel_mse.npy"), allow_pickle=True).item())
    mar_bin_relmse_table = mar_bin_relmse.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    biv_bin_relmse = pd.DataFrame(np.load(os.path.join(save_path, "biv_bin_rel_mse.npy"), allow_pickle=True).item())
    biv_bin_relmse_table = biv_bin_relmse.quantile([0.1, 0.25, 0.5, 0.75, 0.9])

    mar_bin_nbias = pd.DataFrame(np.load(os.path.join(save_path, "mar_bin_nbias.npy"), allow_pickle=True).item())
    mar_bin_nbias_table = mar_bin_nbias.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    mar_bin_nvar = pd.DataFrame(np.load(os.path.join(save_path, "mar_bin_nvar.npy"), allow_pickle=True).item())
    mar_bin_nvar_table = mar_bin_nvar.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    biv_bin_nbias = pd.DataFrame(np.load(os.path.join(save_path, "biv_bin_nbias.npy"), allow_pickle=True).item())
    biv_bin_nbias_table = biv_bin_nbias.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    biv_bin_nvar = pd.DataFrame(np.load(os.path.join(save_path, "biv_bin_nvar.npy"), allow_pickle=True).item())
    biv_bin_nvar_table = biv_bin_nvar.quantile([0.1, 0.25, 0.5, 0.75, 0.9])

    mar_bin_relbias = pd.DataFrame(np.load(os.path.join(save_path, "mar_bin_relbias.npy"), allow_pickle=True).item())
    mar_bin_relbias_table = mar_bin_relbias.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    mar_bin_relvar = pd.DataFrame(np.load(os.path.join(save_path, "mar_bin_relvar.npy"), allow_pickle=True).item())
    mar_bin_relvar_table = mar_bin_relvar.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    biv_bin_relbias = pd.DataFrame(np.load(os.path.join(save_path, "biv_bin_relbias.npy"), allow_pickle=True).item())
    biv_bin_relbias_table = biv_bin_relbias.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    biv_bin_relvar = pd.DataFrame(np.load(os.path.join(save_path, "biv_bin_relvar.npy"), allow_pickle=True).item())
    biv_bin_relvar_table = biv_bin_relvar.quantile([0.1, 0.25, 0.5, 0.75, 0.9])

    # generate latex for cont
    table_to_latex(mar_bin_relmse_table, biv_bin_relmse_table, "relative mean squared error", "binned continuous", float_format = "%.2f", save_mode=save_mode, save_loc=save_loc)
    save_mode = "a"
    # table_to_latex(mar_bin_nbias_table, biv_bin_nbias_table, "normalized bias", "binned continuous", float_format = "%.2f", save_mode=save_mode, save_loc=save_loc, percentage=True)
    # table_to_latex(mar_bin_nvar_table, biv_bin_nvar_table, "normalized variance", "binned continuous", float_format = "%.2f", save_mode=save_mode, save_loc=save_loc)
    # table_to_latex(mar_bin_relbias_table, biv_bin_relbias_table, "relative bias", "binned continuous", float_format = "%.2f", save_mode=save_mode, save_loc=save_loc)
    # table_to_latex(mar_bin_relvar_table, biv_bin_relvar_table, "relative variance", "binned continuous", float_format = "%.2f", save_mode=save_mode, save_loc=save_loc)
