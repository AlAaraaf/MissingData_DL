# load libraries
library(mice)
library(foreach)
library(doParallel)
library(dplyr)

# register cores
cores = as.integer(Sys.getenv("SLURM_CPUS_PER_TASK")) - 1
print("Cores number: ",cores)
cluster = makeCluster(cores)
clusterSetRNGStream(cluster, 9956)
registerDoParallel(cluster)


# preparation
model_name = "cart"
save_name = "house"
complete_filefolder = "complete_0.3_10000"
miss_filefolder = "MCAR_0.3_10000"
save_path = paste("./results/", save_name, "/", miss_filefolder,"/",model_name, sep = '')

#parallel
sample_size = 10
imputed_num = 10

print("Begin Training......")

foreach(i = 0:(sample_size-1), .packages = c("mice"))%dopar%{
  print(paste("Current Sample:",i,"......"))
  current_seed = 42+i
  set.seed(current_seed)
  
  complete_file = paste("./samples/",save_name, '/',complete_filefolder, "/sample_",i,".csv", sep = '')
  missing_file = paste("./samples/",save_name, '/',miss_filefolder, "/sample_",i,".csv", sep = '')
  data_x_i = read.csv(complete_file)
  data_miss_i = read.csv(missing_file)
  
  print("Change variable types......")
  # change categorical variables into factors
  cat_index = 1:(dim(data_x_i)[2]-8)
  num_index = (dim(data_x_i)[2]-7) : dim(data_x_i)[2]
  data_x_i[,cat_index] = lapply(data_x_i[,cat_index], as.factor)
  data_miss_i[,cat_index] = lapply(data_miss_i[,cat_index], as.factor)
  
  print("Data Imputation......")
  # imputation
  data_output_i = mice(data_miss_i, m = imputed_num, method = 'cart', minsplit = 5)
  
  # output results
  print("Output imputed data......")
  for (j in 0:(imputed_num - 1)){
    data_current_imputed = complete(data_output_i, j)
    write.csv(data_current_imputed, paste(save_path, '/imputed_',i,'_',j,'.csv',sep = ''))
  }
  
  print(paste("Sample",i,"done."))
}

stopCluster(cluster)