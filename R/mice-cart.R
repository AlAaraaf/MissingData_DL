rm(list = ls())

# load libraries
library(mice)
library(foreach)
library(doParallel)
library(dplyr)
library(philentropy)
library(caret)
library(readr)

sample_size = 1
imputed_num = 10
model_name = "cart"
complete_filefolder = "complete_0.3_10000"
miss_filefolder = "MCAR_0.3_10000"
index_catenum = cbind(c("boston","credit","house","nhanes",
                        "sim1","sim2","sim3","sim4","sim5"),
                         c(2,9,38,10,6,6,6,6,6))
save_name_list = c("credit","credit","boston","boston",
                   "house","house","nhanes","nhanes",
                   "sim1","sim1","sim2","sim2",
                   "sim3","sim3","sim4","sim4",
                   "sim5","sim5")
lengthdrop_list = c(1,23,1,14,1,46,1,20,1,6,1,6,1,6,1,6,1,6)

for (ii in 1:length(lengthdrop_list)) {
  
  i = 0 
  current_seed = 42+i
  set.seed(current_seed)
  
  save_name = save_name_list[ii]
  lengthdrop = lengthdrop_list[ii]
  categ = index_catenum[which(index_catenum[,1]==save_name),2] %>% as.numeric()
  save_path = paste("../training_data/results/", save_name, "/", miss_filefolder,"/",model_name, sep = '')
  dir.create(save_path, recursive = T)
  
  complete_file = paste("../training_data/samples/",save_name, '/',complete_filefolder, "/sample_",i,"_",1,".csv", sep = '')
  data_complete = read_csv(complete_file, col_names = F, na = "nan") %>% mutate(across(1:categ, as.factor))
  
  missing_file = paste("../training_data/samples/",save_name, '/',miss_filefolder, "/sample_",i,"_",lengthdrop,".csv", sep = '')
  data_miss_i = read_csv(missing_file, col_names = F, na = "nan") %>% mutate(across(1:categ, as.factor))
  
  total_ncol = ncol(data_complete)
  if(categ<total_ncol){
    precessrecall = preProcess(data_complete[,(categ+1):total_ncol], method = "range")
    data_complete[,(categ+1):total_ncol] = predict( precessrecall,data_complete[,(categ+1):total_ncol]) 
    data_miss_i[,(categ+1):total_ncol] = predict( precessrecall,data_miss_i[,(categ+1):total_ncol]) 
  }
  
  
  #### mice imputation 
  data_output_i = mice(data_miss_i, m = imputed_num, 
                       method = 'cart', minbucket = 5, cp=1e-04)
  
  
  ###### evaluation
  
  jsd <- function(p_x, p_y) {
    p_m <- (p_x + p_y) / 2
    (sum(p_x * log(p_x / p_m), na.rm = T) + sum(p_y * log(p_y / p_m), na.rm = T)) / 2
  }
  
  
  metric <- function(x, y, which = c("continuous", "categorical"), from = 0, to = 1) {
    if (which == "continuous") {
      c(mse = mean((x - y) ^ 2),
        mae = mean(abs(x - y)),
        jsd = jsd(density(x, from = from, to = to)$y %>% `/`(sum(.)),
                  density(y, from = from, to = to)$y %>% `/`(sum(.))),
        r2 = cor(x, y) ^ 2)
    } else if (which == "categorical") {
      m <- table(y, x)
      d <- diag(m)
      recall <- d / colSums(m)
      precision <- d / rowSums(m)
      c(acc = sum(d) / sum(m),
        precision = mean(precision, na.rm = T),
        recall = mean(recall, na.rm = T),
        f1 = mean(2 * precision * recall / (precision + recall), na.rm = T),
        jsd = jsd(table(x) / length(x), table(y) / length(y)))
    }
  }
  
  
  na_matrix <- is.na(data_miss_i)
  na_col <- colSums(na_matrix) > 0
  
  result_categ = NULL
  result_numerical = NULL
  for (j in 1:imputed_num ){
    data_prediction = complete(data_output_i, j) 
    res <- mapply(function(y_true, y_pred, is_na) {
      which <- ifelse(is.factor(y_true), "categorical", "continuous")
      if (which == "categorical") y_pred <- factor(y_pred, levels = levels(y_true))
      metric(y_true[is_na], y_pred[is_na], which = which, from = min(y_true), to = max(y_true))
    },
    y_true = data_complete[, na_col, drop = F],
    y_pred = data_prediction[, na_col, drop = F],
    is_na = data.frame(na_matrix[, na_col, drop = F]), SIMPLIFY = F)
    
    result_categ = rbind(result_categ,res[1:categ] %>% bind_rows() %>% colMeans() %>% round(6))
    if(categ<total_ncol){
      result_numerical  = rbind(result_numerical, res[(categ+1):total_ncol] %>% bind_rows() %>% colMeans()) 
    }
  }
  if(lengthdrop==1){
    write.table(colMeans(result_categ), 
                paste(save_path, '/imputed_',lengthdrop,"_",'.csv',sep = ''),
                sep = ',',col.names = FALSE, row.names = FALSE)
  }else{write.table(colMeans(result_categ), 
                    paste(save_path, '/imputed_',lengthdrop,"_","categorical",'.csv',sep = ''),
                    sep = ',',col.names = FALSE, row.names = FALSE)}
  if(categ<total_ncol){
    write.table(colMeans(result_numerical),
                paste(save_path, '/imputed_',lengthdrop,"_","numerical",'.csv',sep = ''),
                sep = ',',col.names = FALSE, row.names = FALSE)
  }
  
}

