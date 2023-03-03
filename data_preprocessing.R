#### change csv file ####

data = read.csv('./data/sim_2.csv')
str(data)

#### change data type for training ####

new_data = data[,2:5]
str(new_data)
new_data[,1:4] = lapply(new_data[,1:4], as.numeric)
write.table(new_data, './data/sim_2.csv', row.names = F, col.names = F, sep = ',')
