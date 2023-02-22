#### change csv file ####

data = read.csv('./data/sim_2.csv')
str(data)

new_data = data[,2:7]
new_data[,1:6] = lapply(new_data[,1:6], as.numeric)
write.table(new_data, './data/sim_2_new.csv', row.names = FALSE, col.names = FALSE, sep = ',')


#### remove 2 variables from the simulation data ####
# the original version of simulation data has 5 explanatory variables
# for calculation ease, remove x4 and x5 from the dataset
data = read.csv('./data/sim_2.csv', header = FALSE)
str(data)

new_data = cbind(data[,1:3], data[6])
str(new_data)
new_data[,1:4] = lapply(new_data[,1:4], as.numeric)
write.table(new_data, './data/sim_2_tiny.csv', row.names = F, col.names = F, sep = ',')
