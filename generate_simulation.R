##### dependency #####
library(dplyr)

##### Covariates comes from MVN(0,1) #####
set.seed(42)
n = 10000
x1 = rnorm(n)
x2 = rnorm(n)
x3 = rnorm(n)
x4 = rnorm(n)
x5 = rnorm(n)
e = rnorm(n)

x = matrix(c(bin_continuous(x1, 2),
             bin_continuous(x2, 3),
             bin_continuous(x3, 5),
             bin_continuous(x4, 8),
             bin_continuous(x5, 10)), nrow = n)

cont.x = matrix(c(x1, x2, x3, x4, x5), nrow = n)
##### wrapped functions #####

## bin response into n bins according to its quantiles
bin_continuous = function(data, num_bins){
  data = data.frame(cont = data)
  new_data = data %>% 
    transmute(disc = cut(cont,
                         breaks = unique(quantile(cont,probs = seq.int(0,1,by = 1/num_bins))),
                         labels = seq(0,num_bins-1, by = 1),
                         include.lowest = TRUE))
  return(as.numeric(levels(new_data$disc))[new_data$disc])
}

## save simulated data
save_file = function(x, y, filename){
  data = cbind(x, y)
  data = data.frame(data, row.names = NULL)
  data = lapply(data, as.numeric)
  write.csv(data, file = filename, row.names = F)
  cat("Saved to",filename, "Finished. \n")
}

##### y1 - x1^2 + exp(X2+x3/3) + sin(x4+x5)+e #####
y = x1^2 + exp(x2 + x3/3) + sin(x4+x5)+e
y = bin_continuous(y, 2)
save_file(x,y, '../training_data/origin/sim1.csv')

##### y2 = x1^2+exp(x2+x3/3)+x4-x5+(0.5+x2^2/2+x5^2/2)*e #####
y = x1^2+exp(x2+x3/3)+x4-x5+(0.5+x2^2/2+x5^2/2)*e
y = bin_continuous(y, 2)
save_file(x,y, '../training_data/origin/sim2.csv')

##### y3 = (5+x1^2/3 + x2^2 + x3^3 +x4+x5)*exp(0.5*e) #####
norm_var1 = rnorm(n,-2,1)
norm_var2 = rnorm(n,2,1)
unif_var = runif(n)
e3 = as.numeric(unif_var < 0.5) * norm_var1 + as.numeric(unif_var >= 0.5) * norm_var2
y = (5+x1^2/3 + x2^2 + x3^3 +x4+x5)*exp(0.5*e3)
y = bin_continuous(y, 2)
save_file(x,y, '../training_data/origin/sim3.csv')

##### y4 = I(u<0.5)*N(-x1, 0.25^2) + I(u>0.5)*N(x1, 0.25^2) #####
unif_var = runif(n)
norm_var = bin_continuous(x1, 3)
y = as.numeric(unif_var <0.5) * rnorm(n, norm_var*-1, 0.25^2) + as.numeric(unif_var >= 0.5) * rnorm(n, norm_var, 0.25^2)
y = bin_continuous(y, 2)
save_file(x,y, '../training_data/origin/sim4.csv')






