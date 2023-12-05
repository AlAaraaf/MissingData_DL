## evaluation 
## MSE, Accuracy, MAE


# preparation
model_name = "cart"
save_name = "sim1"
complete_filefolder = "complete_0.3_10000"
miss_filefolder = "MCAR_0.3_10000"

i=0
complete_file = paste("../training_data/samples/",save_name, '/',complete_filefolder, "/sample_",i,".csv", sep = '')
data_complete = read.csv(complete_file, header = FALSE)

miss_file = paste("../training_data/results/", save_name, "/", miss_filefolder,"/",model_name,"/imputed_",i,".csv", sep = '')
data_miss = read.csv(miss_file, header = FALSE)

nan_file = paste("../training_data/samples/",save_name, '/',miss_filefolder, "/sample_",i,".csv", sep = '')
data_nana = read.csv(nan_file, header = FALSE)

# set NAN level back to NA
data_nana = apply(data_nana, 2, function(x) ifelse(is.nan(x), NA, x))
Nacol = colSums(is.na(data_nana)) > 0
col_to_cp = which(Nacol==T)  %>% as.vector()

MSE = mean((data_complete[,col_to_cp] - data_miss[,col_to_cp])^2)
MAE = mean(abs(data_complete[,col_to_cp] - data_miss[,col_to_cp]))
table_re = table(data_complete[,col_to_cp],data_miss[,col_to_cp])
acc = sum(diag(table_re))/sum(table_re)

eval_rel = data.frame(MSE,MAE,acc)
eval_rel
