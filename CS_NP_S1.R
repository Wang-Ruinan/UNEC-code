# packages
library(mgcv)
library(ggplot2)
library(flexmix)

# --- Functions ---
np_score_lm <- function(df, m_mu, c_val){
  mu <- predict(m_mu, df)       # predicted mean
  if(is.list(mu)){mu <- do.call(cbind,mu)}
  sigma <- summary(m_mu)$sigma  # residual sd
  cdf <- pnorm((c_val - mu) / sigma)
  odds <- cdf / (1 - cdf)
  return(odds)
}

np_score_mix <- function(df, m_mu, c_val){
  mu <- predict(m_mu, df, aggregate=FALSE) # mix model
  if(is.list(mu)){mu <- do.call(cbind,mu)}
  pi <- prior(m_mu) # ratio
  sigma <- parameters(m_mu)["sigma",]
  z1 <- sweep(c_val - mu, 2, sigma, "/")
  z2 <- sweep(pnorm(z1), 2, pi, "*")
  cdf <- rowSums(z2)
  odds <- cdf/(1-cdf)
  return(odds)
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
    data <- data.frame(x=x,y=y)
    
    # --- Data Splitting ---
    idx <- sample(1:n)
    d_train <- data[idx[1:(n/2)], ]
    d_cal   <- data[idx[(n/2+1):(3*n/4)], ]
    d_test  <- data[idx[(3*n/4+1):n], ]
    
    # --- First Stage: Linear prediction model ---
    if(model=="misspecified"){
      model_mu <- lm(y~poly(x,2), data = d_train) # misspecified
    }else{
      model_mu <- flexmix(y~poly(x,2), data = d_train, k=2) # correct
    }
    
    # --- Second Stage: NP scores (R = F(c|x)/(1-F(c|x)) ---
    if(model=="misspecified"){
      d_cal$R <- np_score_lm(d_cal, model_mu, c_threshold)
      d_test$R <- np_score_lm(d_test, model_mu, c_threshold)
    }else{
      d_cal$R <- np_score_mix(d_cal, model_mu, c_threshold)
      d_test$R <- np_score_mix(d_test, model_mu, c_threshold)
    }
    
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
if(model=="misspecified"){
  saveRDS(
    list(
      FDR = results$FDR,
      Power = results$Power
    ),
    file = "NP_misspecified.rds"
  )
}else{
  saveRDS(
    list(
      FDR = results$FDR,
      Power = results$Power
    ),
    file = "NP_correct.rds"
  )
}

cat("--- Selection with NP Scores Results ---\n")
cat("We want to select Y <= ", c_threshold, ".\n")
cat("Alpha:", alpha, "\n")
cat("Replicate:", n_sim, "\n")
cat("Average Empirical FDR:", round(mean(results$FDR), 4), "\n")
cat("Average Empirical Power:", round(mean(results$Power), 4), "\n")