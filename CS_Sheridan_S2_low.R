# packages
library(mgcv)
library(ggplot2)
library(flexmix)
library(MASS)
library(ranger)

# --- Functions ---
generate_data <- function(n) {
  G <- sample(0:3, size = n, replace = TRUE,
              prob = c(0.05, 0.35, 0.25, 0.35))
  W <- rnorm(n)
  
  mu <- rep(NA, n)
  sigma <- rep(NA, n)
  
  mu[G == 0] <- 5.0 + 0.1 * W[G == 0]
  sigma[G == 0] <- 0.5
  
  mu[G == 1] <- 2.4 + 0.1 * W[G == 1]
  sigma[G == 1] <- 3
  
  mu[G == 2] <- 1.0 + 0.1 * W[G == 2]
  sigma[G == 2] <- 0.25
  
  mu[G == 3] <- -0.5 + 0.1 * W[G == 3]
  sigma[G == 3] <- 1.0
  
  Y <- mu + sigma * rnorm(n)
  
  data.frame(G = factor(G), W = W, mu = mu, sigma = sigma, y = Y)
}

calc_score <- function(df, m_mu, m_sigma, c_val) {
  ## estimated mean
  pred_mu <- predict(m_mu, newdata = df)
  
  ## estimated sigma
  #log_var_hat <- as.numeric(predict(m_sigma, newdata = df))
  #pred_sigma <- sqrt(pmax(exp(log_var_hat), 1e-4))
  pred_sigma <- pmax(predict(m_sigma, df), 0.01)
  
  ## Sheridan score
  score <- (pred_mu - c_val) / pred_sigma
  
  return(score)
}

calc_score_t <- function(df, m_mu, sigma_hat_table, c_val) {
  ## estimated mean
  pred_mu <- predict(m_mu, newdata = df)
  
  ## estimated sigma by group
  pred_sigma <- sigma_hat_table[as.character(df$G)]
  pred_sigma <- pmax(pred_sigma, 1e-2)
  
  ## Sheridan score
  score <- (pred_mu - c_val) / pred_sigma
  
  return(score)
}

set.seed(2026)

n <- 5000
c_threshold <- 0
alpha <- 0.1
delta <- 1
n_sim <- 1000

results <- matrix(0, n_sim, 2)
colnames(results) <- c("FDR","Power")
pb <- txtProgressBar(min = 0, max = n_sim, style = 3)

system.time({
  for(sim in 1:n_sim){
    # --- Data Generation ---
    data <- generate_data(n)
    
    # --- Data Splitting ---
    idx <- sample(1:n)
    d_train <- data[idx[1:(n/4)], ]
    d_var   <- data[idx[(n/4+1):(n/2)], ]
    d_cal   <- data[idx[(n/2+1):(3*n/4)], ]
    d_test  <- data[idx[(3*n/4+1):n], ]
    
    # --- First Stage: Linear prediction model ---
    model_mu <- lm(y ~ G+W, data = d_train)
    
    # --- Second Stage: Sigma model ---
    d_var$resid <- d_var$y - predict(model_mu, newdata = d_var)
    #table
    #sigma_hat_table <- tapply(d_var$resid^2, d_var$G, mean)
    #sigma_hat_table <- sqrt(sigma_hat_table)
    
    #sigma_model
    #d_var$log_resid2 <- log(d_var$resid^2 + 1e-4)
    #model_sigma <- lm(log_resid2 ~ G + W, data = d_var[,c("G","W","log_resid2")])
    d_var$abs_resid <- abs(d_var$resid)
    model_sigma <- lm(abs_resid ~ G + W, data = d_var[,c("G","W","abs_resid")])
    
    # --- Third Stage: Sheridan scores (S = (mu - c) / sigma) ---
    #table
    #d_cal$S <- calc_score_t(d_cal, model_mu, sigma_hat_table, c_threshold)
    #d_test$S <- calc_score_t(d_test, model_mu, sigma_hat_table, c_threshold)
    #sigma_model
    d_cal$S <- calc_score(d_cal, model_mu, model_sigma, c_threshold)
    d_test$S <- calc_score(d_test, model_mu, model_sigma, c_threshold)
    
    # --- Fourth Stage: FDR control ---
    # sort by score (descending)
    ord <- order(d_cal$S, decreasing = TRUE)
    S_sorted <- d_cal$S[ord]
    y_sorted <- d_cal$y[ord]
    
    # indicator of false alarm
    false_alarm <- (y_sorted <= c_threshold)
    
    # cumulative sums
    cum_selected <- seq_along(S_sorted)
    cum_false <- cumsum(false_alarm)+delta
    
    # estimated FDR
    est_fdr <- cum_false / pmax(cum_selected, 1) ###
    
    valid <- which(est_fdr <= alpha)
    if(length(valid) == 0){
      best_t <- Inf
    } else {
      best_t <- S_sorted[max(valid)]
    }
    
    # --- Evaluation ---
    selected_idx <- which(d_test$S >= best_t)
    actual_y <- d_test$y[selected_idx] # "y" of selected data
    FDR <- sum(actual_y <= c_threshold) / max(length(selected_idx),1) # test set FDR
    true_idx <- which(d_test$y > c_threshold)
    Power <- sum(actual_y > c_threshold) / max(length(true_idx),1)
    
    results[sim,] <- c(FDR, Power)
    setTxtProgressBar(pb, sim)
  }
})
close(pb)
results <- as.data.frame(results)

## PLOT
df_plot <- rbind(
  data.frame(value = results$FDR, type = "FDR"),
  data.frame(value = results$Power, type = "Power")
)
mean_fdr <- mean(results$FDR)
mean_power <- mean(results$Power)
vlines <- data.frame(
  type = c("FDR", "FDR", "Power"),
  x = c(alpha, mean_fdr, mean_power),
  label = c("alpha", "mean_fdr", "mean_power"),
  color = c("red", "blue", "blue")
)
ggplot(df_plot, aes(x = value)) +
  
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "grey70",
                 color = NA,
                 alpha = 0.5) +
  
  geom_density(linewidth = 1) +
  
  geom_vline(data = vlines,
             aes(xintercept = x, color = color),
             linetype = "dashed",
             linewidth = 1,
             show.legend = FALSE) +
  
  facet_wrap(~type, scales = "free") +
  
  scale_color_identity() +
  
  labs(x = "Value", y = "Density") +
  
  theme_classic()

## SAVE DATA
saveRDS(
  list(
    FDR = results$FDR,
    Power = results$Power
  ),
  file = "Sheridan_sim2.rds"
)

cat("--- Selection with Sheridan Scores Results ---\n")
cat("We want to select Y >= ", c_threshold, ".\n")
cat("Alpha:", alpha, "\n")
cat("Replicate:", n_sim, "\n")
cat("Average Empirical FDR:", round(mean(results$FDR), 4), "\n")
cat("Average Empirical Power:", round(mean(results$Power), 4), "\n")

#data$mu<-2*data$x1+data$x2
#data$sigma<-exp(0.3*data$x3)
#range(data$mu)
#range(data$sigma)