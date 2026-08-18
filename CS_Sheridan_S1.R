# packages
library(mgcv)
library(ggplot2)
library(flexmix)

# --- Functions ---
calc_score <- function(df, m_mu, m_sigma, c_val) {
  if(model=="misspecified"){
    pred_mu <- predict(m_mu, df) # linear model
  }else{
    pred_mu <- unlist(predict(m_mu, df, aggregate=TRUE)) # mix model
  }
  pred_sigma <- pmax(predict(m_sigma, df), 0.01) # 防止分母为0
  #pred_sigma <- sqrt(pmax(predict(m_sigma, df), 0.01))
  return((pred_mu - c_val) / pred_sigma)
}

set.seed(2026)

n <- 5000
c_threshold <- -2
alpha <- 0.1
delta <- 1
n_sim <- 1000
#model <- "misspecified"
model <- "correct"

results <- matrix(0, n_sim, 2)
colnames(results) <- c("FDR","Power")
pb <- txtProgressBar(min = 0, max = n_sim, style = 3)

system.time({
for(sim in 1:n_sim){
  # --- Data Generation ---
  x <- rnorm(n, 0, 1)
  z <- rbinom(n, 1, 0.4)
  mu1 <- -1+x+x^2
  mu2 <- 1-2*x
  y <- z*rnorm(n,mu1,sqrt(2.25)) + (1-z)*rnorm(n,mu2,sqrt(2.25))
  #data <- data.frame(x=x,y=y,sigma=sigma_true)
  #mu_true <- x
  #sigma_true <- ifelse(x < 0, 0.2, 2)
  #y <- mu_true + rnorm(n, sd = sigma_true)
  data <- data.frame(x=x,y=y)
  
  # --- Data Splitting ---
  idx <- sample(1:n)
  d_train <- data[idx[1:(n/4)], ]
  d_var   <- data[idx[(n/4+1):(n/2)], ]
  d_cal   <- data[idx[(n/2+1):(3*n/4)], ]
  d_test  <- data[idx[(3*n/4+1):n], ]
  
  # --- First Stage: Linear prediction model ---
  if(model=="misspecified"){
    model_mu <- lm(y~poly(x,2), data = d_train) # misspecified
  }else{
    model_mu <- flexmix(y~poly(x,2), data = d_train, k=2) # correct
  }
  
  # --- Second Stage: Sigma model ---
  #d_var$res <- abs(d_var$y - predict(model_mu, d_var)) # absolute residuals
  #model_sigma <- gam(res ~ s(x), data = d_var)
  #model_sigma <- lm(res ~ poly(x,3), data = d_var)
  if(model=="misspecified"){
    #d_var$res2 <- (d_var$y - predict(model_mu, d_var))^2 # linear model
    d_var$resid <- d_var$y - predict(model_mu, newdata = d_var)
    d_var$abs_resid <- abs(d_var$resid)
  }else{
    #d_var$res2 <- (d_var$y - unlist(predict(model_mu, d_var,aggregate=TRUE)))^2 # mix model
    d_var$resid <- d_var$y - unlist(predict(model_mu, d_var,aggregate=TRUE))
    d_var$abs_resid <- abs(d_var$resid)
  }
  #model_sigma <- lm(res2 ~ poly(x,3), data=d_var)
  model_sigma <- lm(abs_resid ~ poly(x,3), data = d_var)
  
  # --- Third Stage: Sheridan scores (S = (mu - c) / sigma) ---
  d_cal$S <- calc_score(d_cal, model_mu, model_sigma, c_threshold)
  d_test$S <- calc_score(d_test, model_mu, model_sigma, c_threshold)
  
  # --- Fourth Stage: FDR control ---
  #cal_scores <- sort(unique(d_cal$S), decreasing = FALSE)
  #best_t <- -Inf
  
  #for (t in cal_scores) {
  #  false_alarms_cal <- sum(d_cal$S > t & d_cal$y > c_threshold)
  #  total_selected_cal <- sum(d_cal$S > t)
  #  if (total_selected_cal == 0) next
    
    # estimate FDR
  #  est_fdr <- false_alarms_cal / total_selected_cal
    
  #  if (est_fdr <= alpha) {
  #    best_t <- t
  #    break
  #  } else {
  #    best_t <- t
  #  }
  #}
  
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
  est_fdr <- cum_false / cum_selected
  
  valid <- which(est_fdr <= alpha)
  if(length(valid) == 0){
    best_t <- -Inf
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
if(model=="misspecified"){
  saveRDS(
    list(
      FDR = results$FDR,
      Power = results$Power
    ),
    file = "Sheridan_misspecified.rds"
  )
}else{
  saveRDS(
    list(
      FDR = results$FDR,
      Power = results$Power
    ),
    file = "Sheridan_correct.rds"
  )
}

cat("--- Selection with Sheridan Scores Results ---\n")
cat("We want to select Y <= ", c_threshold, ".\n")
cat("Alpha:", alpha, "\n")
cat("Replicate:", n_sim, "\n")
cat("Average Empirical FDR:", round(mean(results$FDR), 4), "\n")
cat("Average Empirical Power:", round(mean(results$Power), 4), "\n")