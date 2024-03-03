library(RLTquantile)

set.seed(1)
n = 2000
ntrees = 1000
p = 5
x = matrix(runif(n*p, -1, 1), n, p)
var_y = 0.1 + x[,1]^2
y = rnorm(n, 0, sd = sqrt(var_y))
nmin = 50
my_fit = RLT(x, y, model = "quantile", split.gen = "best", mtry = p,
             nmin = nmin, alpha = 0.2, resample.track = TRUE,
             resample.replace = FALSE, resample.prob = 0.5, ntrees = ntrees)

x_test = rbind(seq(0, 0, length.out = p), seq(0, 0, length.out = p))


my_weight = forest.kernel(my_fit, X1 = x,
              vs.train = FALSE)$Kernel / ntrees

my_weight1 = forest.kernel(my_fit, X1 = x, X2 = x,
                          vs.train = FALSE)$Kernel / ntrees

my_weight2 = forest.kernel(my_fit, X1 = x,
                          vs.train = FALSE, OOB = TRUE)$Kernel

my_weight3 = forest.kernel(my_fit, X1 = x_test, X2 = x,
                           vs.train = FALSE, OOB = TRUE)$Kernel

my_weight4 = forest.kernel(my_fit, X1 = x_test, X2 = x,
                           vs.train = FALSE, OOB = FALSE)$Kernel/ntrees

diff2 = (my_weight - my_weight2)
diff3 = (my_weight3- my_weight4)
par(mfrow = c(1, 2))

d1 =  density(x[, 1], weights = my_weight[1, ]/ sum(my_weight[1, ]))
d2 =  density(x[, 5], weights = my_weight[1, ]/ sum(my_weight[1, ]))
plot(d1)
plot(d2)



# regression model
set.seed(1)
n = 1000
p = 5
x = matrix(runif(n*p, -1, 1), n, p)
var_y = 1 + x[,1]^2
y = 10 * x[, 1] + rnorm(n)
nmin = 50
my_fit = RLT(x, y, model = "regression", nmin = 10, nsplit = 10)

x_test = rbind(seq(0, 0.5, length.out = p), seq(0.5, 0, length.out = p))


my_weight = forest.kernel(my_fit, X1 = x_test, X2 = x,
                          vs.train = FALSE)$Kernel

par(mfrow = c(1, 2))

d1 =  density(x[, 1], weights = my_weight[1, ]/ sum(my_weight[1, ]))
d2 =  density(x[, 5], weights = my_weight[1, ]/ sum(my_weight[1, ]))
plot(d1)
plot(d2)


par(mfrow = c(1, 2))
d1 =  density(x[, 1], weights = my_weight[2, ]/ sum(my_weight[2, ]))
d2 =  density(x[, 5], weights = my_weight[2, ]/ sum(my_weight[2, ]))
plot(d1)
plot(d2)

