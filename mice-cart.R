# for local
# setwd('D:/sjx/ISU/projects/DataImputation/MissingData_DL')

# load libraries
library(mice)
library(foreach)
library(doParallel)
library(dplyr)

sample_size = 1
imputed_num = 10

# register cores
cores = min(detectCores()-1, 50)
if (cores > sample_size){
  cores = sample_size
}
print(paste("Cores number: ",cores))
cluster = makeCluster(cores)
clusterSetRNGStream(cluster, 9956)
registerDoParallel(cluster)


# preparation
model_name = "cart"
save_name = "sim_m4"
complete_filefolder = "complete_0.3_5000"
miss_filefolder = "MCAR_0.3_5000"
save_path = paste("../training_data/results/", save_name, "/", miss_filefolder,"/",model_name, sep = '')
dir.create(save_path, recursive = T)


#parallel

print("Begin Training......")

foreach(i = 0:(sample_size-1), .packages = c("mice"))%dopar%{
  print(paste("Current Sample:",i,"......"))
  current_seed = 42+i
  set.seed(current_seed)
  
  missing_file = paste("../training_data/samples/",save_name, '/',miss_filefolder, "/sample_",i,".csv", sep = '')
  data_miss_i = read.csv(missing_file, header = FALSE)
  
  print("Change variable types......")
  # change categorical variables into factors
  cat_index = 1:(dim(data_miss_i)[2]-1)
  data_miss_i[,cat_index] = lapply(data_miss_i[,cat_index], as.factor)
  
  # set NAN level back to NA
  returnNA = function(x){
    levels(x)[levels(x) == 'NaN'] <- NA
    return(x)
  }
  data_miss_i[,cat_index] = lapply(data_miss_i[,cat_index], returnNA)
  
  print("Data Imputation......")
  # imputation
  data_output_i = mice(data_miss_i, m = imputed_num, method = 'cart', minbucket = 5, cp=1e-04)
  
  # output results
  print("Output imputed data......")
  for (j in 1:imputed_num ){
    data_current_imputed = complete(data_output_i, j)
    
    # change categorical variables back to numeric type
    data_current_imputed[,cat_index] = lapply(data_current_imputed[,cat_index], as.numeric)
    
    # remember to ignore row and column names
    write.table(data_current_imputed, paste(save_path, '/imputed_',i,'_',(j-1),'.csv',sep = ''),
              sep = ',',col.names = FALSE, row.names = FALSE)
  }
  print(paste("Sample",i,"done."))
}

stopCluster(cluster)
