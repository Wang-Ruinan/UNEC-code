# packages
library(mgcv)
library(ggplot2)
library(flexmix)
library(MASS)

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

np_score_lm <- function(df, m_mu, sigma_hat_table, c_val){
  mu <- predict(m_mu, df)       # predicted mean
  if(is.list(mu)){mu <- do.call(cbind,mu)}
  sigma <- summary(m_mu)$sigma  # residual sd
  #sigma <- sigma_hat_table[as.character(df$G)]
  #sigma <- pmax(sigma, 1e-2)
  
  cdf <- pnorm((c_val - mu) / sigma)
  odds <- cdf / (1 - cdf)
  return(odds)
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
    d_train <- data[idx[1:(n/2)], ]
    d_cal   <- data[idx[(n/2+1):(3*n/4)], ]
    d_test  <- data[idx[(3*n/4+1):n], ]
    
    # --- First Stage: Linear prediction model ---
    model_mu <- lm(y ~ G+W, data = d_train)
    
    d_train$resid <- d_train$y - predict(model_mu, newdata = d_train)
    sigma_hat_table <- sqrt(tapply(d_train$resid^2, d_train$G, mean))
    
    # --- Second Stage: NP scores (R = F(c|x)/(1-F(c|x)) ---
    d_cal$R <- np_score_lm(d_cal, model_mu, sigma_hat_table, c_threshold)
    d_test$R <- np_score_lm(d_test, model_mu, sigma_hat_table, c_threshold)
    
    # --- Third Stage: FDR control ---
    # sort by score (increasing)
    ord <- order(d_cal$R, decreasing = FALSE)
    R_sorted <- d_cal$R[ord]
    y_sorted <- d_cal$y[ord]
    
    # indicator of false alarm
    false_alarm <- (y_sorted <= c_threshold)
    
    # cumulative sums
    cum_selected <- seq_along(R_sorted)
    cum_false <- cumsum(false_alarm)+delta
    
    # estimated FDR
    est_fdr <- cum_false / cum_selected
    
    valid <- which(est_fdr <= alpha)
    if(length(valid) == 0){
      best_t <- -1
    } else {
      best_t <- R_sorted[max(valid)]
    }
    
    # --- Evaluation ---
    selected_idx <- which(d_test$R <= best_t)
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
  file = "NP_sim2.rds"
)

cat("--- Selection with NP Scores Results ---\n")
cat("We want to select Y <= ", c_threshold, ".\n")
cat("Alpha:", alpha, "\n")
cat("Replicate:", n_sim, "\n")
cat("Average Empirical FDR:", round(mean(results$FDR), 4), "\n")
cat("Average Empirical Power:", round(mean(results$Power), 4), "\n")
