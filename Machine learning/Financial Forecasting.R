library(readxl)
library(dplyr)
library(neuralnet)
library(caret)
library(Metrics)
library(ggplot2)

data <- read_excel("D:/Uni work/University/Year 2/Machine learning/ExchangeUSD.xlsx", col_types = c("skip", "skip", "numeric"))

if(nrow(data) == 0) stop("Exchange rates data is empty.")

exchange_rates <- data$`USD/EUR`
training_set <- exchange_rates[1:400]
testing_set <- exchange_rates[401:500]

min_rate <- min(training_set)
max_rate <- max(training_set)
normalize <- function(x) (x - min_rate) / (max_rate - min_rate)
denormalize <- function(x) x * (max_rate - min_rate) + min_rate

normalized_training <- normalize(training_set)
normalized_testing <- normalize(testing_set)

create_AR_data <- function(data, lag = 1) {
  max_index <- length(data)
  if (max_index <= lag) stop("Not enough data to create AR matrix with specified lag.")
  X <- lapply(1:lag, function(l) data[(lag + 1 - l):(max_index - l)])
  X <- do.call(cbind, X)
  colnames(X) <- paste("Lag", 1:lag, sep = "")
  y <- data[(lag + 1):max_index]
  return(list(X = X, y = y))
}

lags <- c(1, 2, 3, 4, 5)
nn_configs <- list(
  list(hidden = 5),         # 1 layer; 5 neurons
  list(hidden = c(3, 2)),   # 2 layers; 3 and 2 neurons
  list(hidden = 10)         # 1 layer; 10 neurons
)

results <- data.frame(Lag = integer(), Config = character(), RMSE = numeric(), MAE = numeric(), stringsAsFactors = FALSE)

for (lag in lags) {
  AR_data <- create_AR_data(normalized_training, lag)
  for (config in nn_configs) {
    set.seed(123)
    formula <- as.formula(paste("y ~", paste(colnames(AR_data$X), collapse = " + ")))
    nn_model <- neuralnet(formula, data = data.frame(AR_data$X, y = AR_data$y), hidden = config$hidden, linear.output = TRUE, threshold = 0.01)
    AR_test_data <- create_AR_data(normalized_testing, lag)
    predictions <- neuralnet::compute(nn_model, AR_test_data$X)$net.result
    predictions <- denormalize(predictions)
    actuals <- denormalize(testing_set[(lag + 1):length(testing_set)])
    
    rmse_value <- rmse(actuals, predictions)
    mae_value <- mae(actuals, predictions)
    
    results <- rbind(results, data.frame(Lag = lag, Config = toString(config$hidden), RMSE = rmse_value, MAE = mae_value))
  }
}

print(results)

ggplot(results, aes(x = Lag, y = RMSE, color = Config)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  ggtitle("Performance of Different MLP Configurations")
