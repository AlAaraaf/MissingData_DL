import numpy as np

def load_dataset(dataset: str, 
                 miss_mechanism: str, 
                 missing_rate: str, 
                 sample_size: str, 
                 sample_id: str):
    
    # check whether the dataset file exist or not
    if dataset not in datalevel.keys():
        print("Wrong dataset: {}".format(dataset))
        exit

    datalevel = np.load('datalevel.npy', allow_pickle='TRUE').item()
    all_levels_dict = datalevel[dataset]
    file_name = '../training_data/samples/{}/{}_{}_{}/sample_{}.csv'.format(dataset, miss_mechanism, missing_rate, sample_size, sample_id)  
    data_x_i = np.loadtxt('../training_data/samples/{}/complete_{}_{}/sample_{}.csv'.format(dataset, missing_rate, sample_size,sample_id), delimiter=",").astype(np.float32)

    miss_data_x = np.loadtxt(file_name, delimiter=",").astype(np.float32)
    data_m = 1 - np.isnan(miss_data_x).astype(np.float32)
    
    return  (data_x_i, miss_data_x, data_m, all_levels_dict)
        