# packages
library(mgcv)
library(ggplot2)
library(flexmix)

# --- Functions ---
generate_data_mix <- function(n) {
  x <- rnorm(n, 0, 1)
  
  ## x-dependent mixture probability
  pi_x <- plogis(4 * (x - 0.25))
  
  z <- rbinom(n, 1, pi_x)
  
  ## high-mean, high-uncertainty component
  mu_H <- 2.7 + 0.5 * x + 0.2 * x^2
  sigma_H <- 3.0
  
  ## stable positive component for smaller x
  mu_L <- 0.7 - 1.4 * x
  sigma_L <- 0.30
  
  y <- ifelse(
    z == 1,
    rnorm(n, mean = mu_H, sd = sigma_H),
    rnorm(n, mean = mu_L, sd = sigma_L)
  )
  
  ## true conditional mean and sd, useful for diagnosis
  cond_mu <- pi_x * mu_H + (1 - pi_x) * mu_L
  cond_var <- pi_x * (sigma_H^2 + mu_H^2) +
    (1 - pi_x) * (sigma_L^2 + mu_L^2) -
    cond_mu^2
  cond_sd <- sqrt(pmax(cond_var, 1e-8))
  
  data.frame(
    x = x,
    z_latent = z,
    pi_x = pi_x,
    mu_H = mu_H,
    mu_L = mu_L,
    cond_mu = cond_mu,
    cond_sd = cond_sd,
    y = y
  )
}

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

np_score <- function(df, m_mu, m_sigma, c_val){
  mu <- predict(m_mu, df)       # predicted mean
  if(is.list(mu)){mu <- do.call(cbind,mu)}
  #res2
  #sigma <- sqrt(pmax(predict(m_sigma, df), 0.01))
  #abs_res
  sigma <- pmax(predict(m_sigma, df), 0.01)
  cdf <- pnorm((c_val - mu) / sigma)
  odds <- cdf / (1 - cdf)
  return(odds)
}

set.seed(2026)

n <- 5000
c_threshold <- 1
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
    data <- generate_data_mix(n)
    
    # --- Data Splitting ---
    idx <- sample(1:n)
    d_train <- data[idx[1:(n/2)], ]
    d_cal   <- data[idx[(n/2+1):(3*n/4)], ]
    d_test  <- data[idx[(3*n/4+1):n], ]
    
    # --- First Stage: Linear prediction model ---
    if(model=="misspecified"){
      model_mu <- lm(y~poly(x,2), data = d_train) # misspecified
      #res2
      #d_train$res2 <- (d_train$y - predict(model_mu, d_train))^2 # linear model
      #model_sigma <- lm(res2 ~ poly(x,3), data=d_train)
      #abs_res
      #d_train$resid <- d_train$y - predict(model_mu, newdata = d_train)
      #d_train$abs_resid <- abs(d_train$resid)
      #model_sigma <- lm(abs_resid ~ poly(x,3), data = d_train)
    }else{
      #model_mu <- flexmix(y~poly(x,2), data = d_train, k=2) # correct
      model_mu <- flexmix(y ~ poly(x, 2, raw = TRUE), data = d_train, k = 2, concomitant = FLXPmultinom(~ x))
    }
    
    # --- Second Stage: NP scores (R = F(c|x)/(1-F(c|x)) ---
    if(model=="misspecified"){
      #location-shift
      d_cal$R <- np_score_lm(d_cal, model_mu, c_threshold)
      d_test$R <- np_score_lm(d_test, model_mu, c_threshold)
      #location-scale
      #d_cal$R <- np_score(d_cal, model_mu, model_sigma, c_threshold)
      #d_test$R <- np_score(d_test, model_mu, model_sigma, c_threshold)
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
    file = "NP_misspecified_sim4.rds"
  )
}else{
  saveRDS(
    list(
      FDR = results$FDR,
      Power = results$Power
    ),
    file = "NP_correct_sim4.rds"
  )
}

cat("--- Selection with NP Scores Results ---\n")
cat("We want to select Y <= ", c_threshold, ".\n")
cat("Alpha:", alpha, "\n")
cat("Replicate:", n_sim, "\n")
cat("Average Empirical FDR:", round(mean(results$FDR), 4), "\n")
cat("Average Empirical Power:", round(mean(results$Power), 4), "\n")
