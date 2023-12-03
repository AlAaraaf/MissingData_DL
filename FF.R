library(magrittr)
library(keras)
library(caret)
library(dplyr)

# preparation
model_name = "FF"
save_name = "boston"
complete_filefolder = "complete_0.3_10000"
miss_filefolder = "MCAR_0.3_10000"
save_path = paste("../training_data/results/", save_name, "/", miss_filefolder,"/",model_name, sep = '')
dir.create(save_path, recursive = T)

i=0
missing_file = paste("../training_data/samples/",save_name, '/',miss_filefolder, "/sample_",i,".csv", sep = '')
data_miss_i = read.csv(missing_file, header = FALSE)


### split date into two part: inlcude NA and no NA
data = apply(data_miss_i, 2, function(x) ifelse(is.nan(x), NA, x))
data_miss_i = data.frame(ID= c(1:nrow(data_miss_i)),data)
Nacol = colSums(is.na(data_miss_i)) > 0

index = complete.cases(data_miss_i)
train_data = data_miss_i[index==T,]
test_data = data_miss_i[index==F,]


x_train = train_data[,-c(1,which(Nacol==T))] %>% as.matrix()
y_train = as.numeric(train_data[,which(Nacol==T)])

x_test = test_data[,-c(1,which(Nacol==T))] %>% as.matrix()
#y_test = as.numeric(test_data[,2])

## or do the center and scale
preProc <- preProcess(x_train, method = c("center", "scale"))
train_x_preprocess = as.matrix(predict(preProc, x_train))
test_x_preprocess = as.matrix(predict(preProc, x_test))

#### build model
build_convnet = function(unit_list, input_shape, batch_normalization = F, optimizer_pick = F,
                         initial_learning_rate = 0.1, printmodel = T){
  model <- keras_model_sequential() 
  model %>% layer_flatten(input_shape = input_shape)
    for(unit in unit_list){
      if (batch_normalization){model %>% layer_dense(units = unit, activation = 'relu') %>% layer_batch_normalization()
      }else {model %>% layer_dense(units = unit, activation = 'relu')}
    }
  model %>% layer_dense(1, activation = 'sigmoid')
  if (printmodel) summary(model, show_trainable = T)
  
  optimizer_pick <- ifelse(optimizer_pick, optimizer_adam, optimizer_rmsprop)
  learning_rate <- learning_rate_schedule_exponential_decay(initial_learning_rate,
                                                            decay_steps = 5, decay_rate = 0.9, staircase = T)
  model %>% compile(
    loss = loss_binary_crossentropy(),
    optimizer = optimizer_pick(learning_rate = learning_rate),
    metrics = list("acc")
  )
  model
}

## Training our model
unit_list <- list(2 ^ (3:4), 2 ^ (3:5), 2 ^ (3:6), 2 ^ (3:7),
                  2 ^ (4:5), 2 ^ (4:6), 2 ^ (4:7), 2 ^ (4:8),
                2 ^ (5:6), 2 ^ (5:7), 2 ^ (5:8), 2 ^ (5:9))
pars <- expand.grid(lr_initial = c(0.001,0.01), unit = unit_list ,optimizer_pick = c(T,F))
res <- NULL
for (kk in 1:nrow(pars)) {
  lr_initial = pars$lr_initial[kk]
  unit = pars$unit[[kk]]
  optimizer = pars$optimizer_pick[[kk]]
  
  model <- build_convnet(unit_list = unit, input_shape = ncol(x_train), 
                         batch_normalization = T  , optimizer_pick = optimizer,
                         initial_learning_rate = lr_initial, printmodel = T)
  
  callbacks <- callback_early_stopping(patience = 10, restore_best_weights = T)
  history <- model %>%
    fit(x_train, y_train, epochs = 100, batch_size = 32, verbose = 1,
        callbacks = callbacks, validation_split = .5)
  res <- rbind(res, last(data.frame(history$metrics)))
}

### chose best model to predict for test data
results = cbind(pars, res)
results

lr_initial = pars$lr_initial[which.max(results$val_acc)]
unit = pars$unit[[which.max(results$val_acc)]]
optimizer = pars$optimizer_pick[[which.max(results$val_acc)]]
best_model <- build_convnet(unit_list = unit, input_shape = ncol(x_train), 
                       batch_normalization = T  , optimizer_pick = optimizer,
                       initial_learning_rate = lr_initial, printmodel = T)

result = predict(best_model, x_test)
result[result<=0.5] = 0
result[result>=0.5] = 1
test_data[,which(Nacol==T)] = result

## combine test data and train in original order
data_result  =  rbind(train_data,test_data)
data = data_result[order(data_result$ID, decreasing = F), ]


###  save data
write.table(data[,-1], paste(save_path, '/imputed_',i,'_',0,'.csv',sep = ''),
              sep = ',',col.names = FALSE, row.names = FALSE)

