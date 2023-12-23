
rm(list = ls())


library(tidyverse)
library(reticulate)
library(tensorflow)
library(keras)
library(caret)
library(philentropy)

### the functions for our data precess
add_na <- function(data, missing_rate = NULL, missing_idx = NULL) {
  if (is.null(missing_idx)) {
    if (length(missing_rate) != ncol(data)) {
      if (length(missing_rate) == 1) missing_rate <- rep(missing_rate, ncol(data))
      else stop("missing_rate should have length of 1 or ncol(data).")
    }
    idx_missing <- rbinom(n = prod(dim(data)), size = 1, prob = rep(missing_rate, each = nrow(data)))
    missing_idx <- `dim<-`(as.logical(idx_missing), dim(data))
  }
  data[missing_idx] <- NA
  data
}

preprocess_data <- function(data, model = NULL, which = c("continuous", "categorical"), cts_method = c("center", "scale")) {
  if (is.null(model)) model <- list(continuous = preProcess(data, method = cts_method),
                                    categorical = dummyVars(~ ., data))
  model <- model[which]
  list(model = model, data = as_tibble(Reduce(f = function(x, y) data.frame(predict(y, x)), x = model, init = data)))
}

##### build model for all avriable missing
build_model <- function(unit_list, input_shape, output_shape,
                        learning_rate = 0.1, batch_normalization = F, regularizer = "none",
                        printmodel = T) {
  k_clear_session()

  hidden <- input <- layer_input(shape = input_shape)

  for (unit in unit_list) {
    kernel_regularizer <- switch(regularizer, none = NULL, l1 = regularizer_l1(), l2 = regularizer_l2())
    hidden <- layer_dense(units = unit, activation = "leaky_relu", kernel_regularizer = kernel_regularizer)(hidden)
    if (batch_normalization) hidden <- layer_batch_normalization()(hidden)
  }

  output_nm <- names(output_shape)
  output <- lapply(seq_along(output_shape), function(i) {
    nclass <- output_shape[i]
    layer_dense(units = nclass, activation = ifelse(nclass > 1, "softmax", "linear"), name = output_nm[i])(hidden)
  }) %>% setNames(output_nm)
  output <- do.call(layer_concatenate, unname(output))

  model <- keras_model(inputs = input, outputs = output)
  if (printmodel) summary(model, show_trainable = T)

  # loss <- lapply(output_shape, function(nclass) {
  #   if (nclass > 1) loss_categorical_crossentropy() else loss_mean_squared_error()
  # })

  custom_loss <- function(y_true, y_pred) {
    groups <- if (length(output_shape) == 1) 1L else as.integer(output_shape)
    axis <- -1L
    y_true_list <- tf$split(y_true, groups, axis = axis)
    y_pred_list <- tf$split(y_pred, groups, axis = axis)
    losses <- lapply(seq_along(output_shape), function(i) {
      nclass <- output_shape[i]
      loss_fun <- if (nclass > 1) loss_categorical_crossentropy() else loss_mean_squared_error()
      loss_fun(y_true_list[[i]], y_pred_list[[i]])
    })
    tf$math$add_n(losses)  # tf$reduce_sum(losses, axis = -1L)
  }

  model %>% compile(
    loss = custom_loss,
    optimizer = keras$optimizers$legacy$Adam(learning_rate = learning_rate)
    # metrics = lapply(output_shape, function(x) NULL)
    # vector("list", length(output_nm)) %>% setNames(output_nm)
    # metrics = list("acc", "AUC", "Precision", "Recall")
  )
  model
}


#### build model for one variable missing
build_one_model <- function(unit_list, input_shape,output_shape,
                            learning_rate_initial = 0.1,
                             batch_normalization = F,
                             learning_rate_decay = T, regularizer = "none", printmodel = T) {
  k_clear_session()
  hidden <- input <- layer_input(shape = input_shape)
  for (unit in unit_list) {
    kernel_regularizer <- switch(regularizer, none = NULL, l1 = regularizer_l1(), l2 = regularizer_l2())
    hidden <- layer_dense(units = unit, activation = "relu", kernel_regularizer = kernel_regularizer)(hidden)
    if (batch_normalization) hidden <- layer_batch_normalization()(hidden)
  }
  output <- lapply(output_shape, function(nclass) layer_dense(units = nclass, activation = "softmax")(hidden))
  model <- keras_model(inputs = input, outputs = output)
  if (printmodel) summary(model, show_trainable = T)
  
  learning_rate <- if (learning_rate_decay) {
    learning_rate_schedule_exponential_decay(learning_rate_initial,
                                             decay_steps = 5, decay_rate = 0.9, staircase = T)
  } else learning_rate_initial
  
  model %>% compile(
    loss = replicate(length(output_shape), loss_categorical_crossentropy()),
    # keras$losses$CategoricalFocalCrossentropy()
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = list("acc", "AUC", "Precision", "Recall")
  )
  model
}


##### pre data
sample_size = 1
imputed_num = 10
model_name = "FF"
complete_filefolder = "complete_0.3_10000"
miss_filefolder = "MCAR_0.3_10000"
index_catenum = cbind(c("boston","credit","house","nhanes","sim1","sim2","sim3","sim4"),c(2,9,38,10,6,6,6,6))
### boston: 14; credit:23; house:46; nhanes:20

save_name_list = c("credit","credit","boston","boston",
                   "house","house","nhanes","nhanes",
                   "sim1","sim1","sim2","sim2",
                   "sim3","sim3","sim4","sim4")
lengthdrop_list = c(1,23,1,14,1,46,1,20,1,6,1,6,1,6,1,6)


for(ii in 1:16){
  ii= 16
  save_name = save_name_list[ii]
  lengthdrop = lengthdrop_list[ii]
  save_path = paste("../training_data/results/", save_name, "/", miss_filefolder,"/",model_name, sep = '')
  dir.create(save_path, recursive = T)
  i = 0
  current_seed = 42+i
  set.seed(current_seed)
  
  missing_file = paste("../training_data/samples/",save_name, '/',miss_filefolder, "/sample_",i,"_",lengthdrop,".csv", sep = '')
  data_miss_i = read.csv(missing_file, header = FALSE, na.strings = "nan")
  complete_file = paste("../training_data/samples/",save_name, '/',complete_filefolder, "/sample_",i,"_",1,".csv", sep = '')
  data_complete = read.csv(complete_file, header = FALSE)
  
  column_to_compare = which(colSums(is.na(data_miss_i)) > 0)
  total_ncol  = ncol(data_miss_i)
  
  
  if(length(column_to_compare)==1){
    missing_rate <- c(0.3,rep(0,(ncol(data_miss_i)-1)))
  }else{missing_rate <- 0.3}
  
  
  categ = index_catenum[which(index_catenum[,1]==save_name),2] %>% as.numeric()
  if(categ<total_ncol){col_type <- list(categorical = 1:categ, continuous = (1+categ):ncol(data_complete))
  }else{col_type <- list(categorical = 1:categ, continuous = NULL)}
  
  data_train_early <- data_complete %>%
    mutate(across(col_type$categorical, as.factor),
           across(col_type$continuous, as.numeric))
  
  data_test <- data_miss_i %>%
    mutate(across(col_type$categorical, as.factor),
           across(col_type$continuous, as.numeric))
  
  if(length(column_to_compare)>1){
    
    data_test[,1] = factor(data_test[,1],
                           labels = letters[1:length(unique(data_train_early[,1]))],
                           exclude = NA)
    data_train_early[,1] = factor(data_train_early[,1],
                                  labels = letters[1:length(unique(data_train_early[,1]))])
    
    data_train <- add_na(data_train_early, missing_rate)
    dt_missing <-  list(train = data_train, test = data_test)
    dt <- list(train = data_train_early, test = data_train_early)
    
    preProc <- preprocess_data(dt_missing$train, cts_method = "range")
    dt_ready_x <- list(train = preProc$data %>% slice_sample(prop = 1),
                       test = preprocess_data(dt_missing$test, model = preProc$model)$data) %>%
      lapply(function(x) mutate(x, across(everything(), ~ replace_na(., 0L))))
    dt_ready_y <- lapply(dt, function(x) preprocess_data(x, model = preProc$model)$data)
    
    
    
    ## Training our model
    pars <- expand.grid(lr_initial = 10 ^ (-2:-3), unit = lapply(4:7, function(x) 2 ^ (3:x)))
    res <- NULL
    early_stop <- F
    lr_decay <- F
    
    # for (kk in 1:nrow(pars)) {
    kk <- 5
    lr_initial <- pars$lr_initial[[kk]]
    unit <- pars$unit[[kk]]
    lr_initial <- 1e-4
    # unit <- c(64, 32, 16, 32, 64)
    # unit <- c(32, 16, 32)
    unit <- c(16, 8, 16)
    # unit <- c(8, 4, 8)
    
    nlvl <- dt$train %>% sapply(nlevels) %>% replace(. == 0, 1L)
    model <- build_model(unit_list = unit, input_shape = ncol(dt_ready_x$train), output_shape = nlvl,
                         learning_rate = lr_initial, printmodel = T)
    # model$loss
    # model$output
     keras$utils$plot_model(model, "model.png", show_shapes = T, expand_nested = T, show_layer_activations = T, dpi = 300L)
    
    callbacks <- if (early_stop) callback_early_stopping(patience = 30, restore_best_weights = T) else NULL
    callbacks <- callback_reduce_lr_on_plateau(factor = 0.5, patience = 10)
    # weight <- as.list(length(y_train) / table(y_train) / 2)
    weight <- NULL
    
    
    history <- model %>%
      fit(x = dt_ready_x$train %>% as.matrix,
          y = dt_ready_y$train %>% as.matrix, # tf$split(dt_ready_y$train, nlvl, axis = 1L) %>% setNames(names(nlvl)),
          batch_size = 1024, epochs = 100, class_weight = weight,
          callbacks = callbacks, validation_split = .2, verbose = 1)
    #   res <- rbind(res, last(data.frame(history$metrics)))
    # }
    
    ### chose best model to predict for test data
    # results = cbind(pars, res)
    # results
    
    # lr_initial = pars$lr_initial[which.max(results$val_loss)]
    # unit = pars$unit[[which.max(results$val_loss)]]
    # model <- build_model(unit_list = unit, input_shape = ncol(dt_ready_x$train), output_shape = nlvl,
    #                           learning_rate = lr_initial, printmodel = T)
    # callbacks <- if (early_stop) callback_early_stopping(patience = 30, restore_best_weights = T) else NULL
    # callbacks <- callback_reduce_lr_on_plateau(factor = 0.5, patience = 10)
    # # weight <- as.list(length(y_train) / table(y_train) / 2)
    # weight <- NULL
    # history <- model %>%
    #   fit(x = dt_ready_x$train %>% as.matrix,
    #       y = dt_ready_y$train %>% as.matrix, # tf$split(dt_ready_y$train, nlvl, axis = 1L) %>% setNames(names(nlvl)),
    #       batch_size = 1024, epochs = 100, class_weight = weight,
    #       callbacks = callbacks, validation_split = .2, verbose = 1)
    
    result <- evaluate(model,
                       x = dt_ready_x$test %>% as.matrix,
                       y = dt_ready_y$test %>% as.matrix  # tf$split(dt_ready_y$test, nlvl, axis = 1L) %>% setNames(names(nlvl))
    )
    result
    
    pred_test <- predict(model, dt_ready_x$test %>% as.matrix)
    data_test_predict <- sapply(if (is.matrix(pred_test)) tf$split(pred_test, nlvl, axis = 1L) else pred_test,
                                function(x) as.vector(if (ncol(x) > 1) keras$backend$argmax(x) else x)) %>%
      `colnames<-`(colnames(dt$test)) %>% as_tibble
    data_test_true <- preprocess_data(dt$test, model = preProc$model, which = c("continuous"))$data
    # data_test_predict
    # data_test_true
    tempdata = data_test_true[,1]
    data_test_true[,1] = factor(tempdata[[1]],
                                labels = seq(0,length(unique(tempdata[[1]]))-1))
    
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
    
    res <- mapply(function(y_true, y_pred, is_na) {
      which <- ifelse(is.factor(y_true), "categorical", "continuous")
      if (which == "categorical") y_pred <- factor(y_pred, levels = levels(y_true))
      metric(y_true[is_na], y_pred[is_na], which = which, from = min(y_true), to = max(y_true))
    },
    y_true = data_test_true[, na_col, drop = F],
    y_pred = data_test_predict[, na_col, drop = F],
    is_na = data.frame(na_matrix[, na_col, drop = F]), SIMPLIFY = F)
    
    
    
    result_categ = rbind(result_categ,res[1:categ] %>% bind_rows() %>% colMeans() %>% round(6))
    if(categ<total_ncol){
      result_numerical  = rbind(result_numerical, res[(categ+1):total_ncol] %>% bind_rows() %>% colMeans()) 
    }
    
    if(lengthdrop==1){
      write.table(colMeans(result_categ), 
                  paste(save_path, '/imputed_',lengthdrop,'.csv',sep = ''),
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
  
  if(length(column_to_compare)==1){
    ### split date into two part: include NA and no NA
    idx_shuffle <- sample(1:nrow(data_miss_i))
    data_miss_i_shuffle <- data_miss_i[idx_shuffle, ]
    Nacol <- colSums(is.na(data_miss_i_shuffle)) > 0
    idx_complete <- complete.cases(data_miss_i_shuffle)
    train_data <- data_miss_i_shuffle[idx_complete, ]
    test_data <- data_miss_i_shuffle[!idx_complete, ]
    
    x_train <- train_data[, !Nacol, drop = F] %>% as.matrix()
    y_train <- train_data[, Nacol, drop = F] %>% as.matrix() %>% apply(2, to_categorical, simplify = F)
    x_test <- test_data[, !Nacol, drop = F] %>% as.matrix()
    y_test <- test_data[, Nacol, drop = F] %>% as.matrix()
    
    ## or do the center and scale
    
    total_ncol = ncol(data_miss_i)
    if(categ<total_ncol){
      precessrecall = preProcess(data_complete[,(categ+1):total_ncol], method = "range")
      x_train[,categ:(total_ncol-1)] = predict( precessrecall,x_train[,categ:(total_ncol-1)]) 
      x_test[,categ:(total_ncol-1)] = predict( precessrecall,x_test[,categ:(total_ncol-1)]) 
    }
    
    pars <- expand.grid(lr_initial = 10 ^ (-2:-4), unit = lapply(4:7, function(x) 2 ^ (3:x)))
    res <- NULL
    early_stop <- F
    lr_decay <- F
    
    
     for (kk in 1:nrow(pars)) {
    kk = 5
    lr_initial <- pars$lr_initial[[kk]]
    unit <- pars$unit[[kk]]
    lr_initial = 1e-2
    
    model <- build_one_model(unit_list = unit,input_shape = ncol(x_train), output_shape = sapply(y_train, ncol),
                             learning_rate_initial = lr_initial, 
                             batch_normalization = F,
                             learning_rate_decay = lr_decay,
                             regularizer = "none", printmodel = T)
    model$loss
    model$output
    keras$utils$plot_model(model, "model.pdf", show_shapes = T, expand_nested = T, show_layer_activations = T, dpi = 300L)
    
    callbacks <- if (early_stop) callback_early_stopping(patience = 10, restore_best_weights = T) else NULL
    # weight <- as.list(length(y_train) / table(y_train) / 2)
    weight <- NULL
    
    history <- model %>%
      fit(#x = reticulate::r_to_py(train_x_preprocess), y = reticulate::r_to_py(y_train),
        x = x_train, y = y_train,
        batch_size = 256, epochs = 100, class_weight = weight,
        callbacks = callbacks, validation_split = .2, verbose = 1)
     res <- rbind(res, last(data.frame(history$metrics)))
    }
    
    ### chose best model to predict for test data
     results = cbind(pars, res)
     results
    #
    # lr_initial = pars$lr_initial[which.max(results$val_acc)]
    # unit = pars$unit[[which.max(results$val_acc)]]
    # optimizer = pars$optimizer_pick[[which.max(results$val_acc)]]
    # best_model <- build_convnet(unit_list = unit, input_shape = ncol(x_train),
    #                             batch_normalization = T  , optimizer_pick = optimizer,
    #                             initial_learning_rate = lr_initial, printmodel = T)
    
    # best_model = model
    
    
    test_y_true <- data_complete[idx_shuffle, ][!idx_complete, Nacol, drop = F] %>% as.matrix() %>% apply(2, to_categorical, simplify = F)
    result <- evaluate(model, x = x_test, y = test_y_true)
    result
    result = rbind(result[2],result[4],result[5],2*result[4]*result[5]/(sum(result[4:5])) )
    test_y_predict <- predict(model, x_test)
    
    
    jsd <- function(p_x, p_y) {
      p_m <- (p_x + p_y) / 2
      (sum(p_x * log(p_x / p_m), na.rm = T) + sum(p_y * log(p_y / p_m), na.rm = T)) / 2
    }
    
    x = factor(apply(test_y_true[[1]],1,which.max)-1, levels = levels(data_train_early[,column_to_compare]))
    y = factor(apply(test_y_predict[[1]],1,which.max)-1, levels = levels(data_train_early[,column_to_compare]))
    jsd = jsd(table(x) / length(x), table(y) / length(y))
    
    write.table(rbind(result,jsd), 
                paste(save_path, '/imputed_',lengthdrop,"_","auto",'.csv',sep = ''),
                sep = ',',col.names = FALSE, row.names = FALSE)
    
    m <- table(x, y)
    d <- diag(m)
    recall <- d / colSums(m)
    precision <- d / rowSums(m)
    acc = sum(d) / sum(m)
    precision = mean(precision, na.rm = T)
    recall = mean(recall, na.rm = T)
    f1 = mean(2 * precision * recall / (precision + recall), na.rm = T)
    write.table(rbind(acc,precision, recall, f1,jsd), 
                paste(save_path, '/imputed_',lengthdrop,'.csv',sep = ''),
                sep = ',',col.names = FALSE, row.names = FALSE)
    
  }
  
  
}

