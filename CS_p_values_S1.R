# 加载必要的库
library(mgcv) # 用于拟合平滑的均值和方差模型
library(ggplot2)
library(flexmix)

# --- Functions ---
v_score <- function(df, df_y, m_mu){
  #pred_mu <- predict(m_mu, df) # linear
  pred_mu <- unlist(predict(m_mu, df, aggregate=TRUE)) # mix model
  return(M*(df_y>c_threshold)-pred_mu)
}

p_values <- function(df, u_value, v_value){
  n <- length(v_value)
  F_hat <- ecdf(v_value)
  p <- (n * F_hat(df$V) + u_value) / (n + 1)
  return(p)
}

#p_values <- function(df, u_value, v_value){
#  n <- length(v_value)
#  rank_val <- sapply(df$V, function(v) sum(v_value < v))
#  tie_val  <- sapply(df$V, function(v) sum(v_value == v))
#  p <- (rank_val + (1 + tie_val) * u_value) / (n + 1)
#  return(p)
#}

BH_procedure <- function(p, alpha){
  
  m <- length(p)
  p_sorted <- sort(p) # increasing
  index_sorted <- order(p) # small to big
  
  threshold <- (1:m)/m * alpha
  
  k <- max(which(p_sorted <= threshold), 0)
  
  selected <- rep(FALSE, m)
  
  if (k > 0) {
    selected[index_sorted[1:k]] <- TRUE
  }
  
  return(list(
    selected = selected,
    k = k,
    cutoff = ifelse(k>0, p_sorted[k], NA)
  ))
}

set.seed(2026)

n <- 5000
c_threshold <- -2
alpha <- 0.1
M <- 100
n_sim <- 1000
model <- "misspecified"
#model <- "correct"

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
    model_mu <- lm(y~poly(x,2), data = d_train) # linear
  }else{
    model_mu <- flexmix(y~poly(x,2), data = d_train, k=2) # correct
  }
  
  # --- Second Stage: Nonconfirmity scores ---
  d_cal$V <- v_score(d_cal, d_cal$y, model_mu)
  d_test$V <- v_score(d_test, c_threshold, model_mu)
  
  # --- Third Stage: Conformal p-values ---
  U <- runif(nrow(d_test))
  d_test$p <- p_values(d_test, U, d_cal$V)
  
  # --- Fourth Stage: BH procedure ---
  bh <- BH_procedure(d_test$p, alpha)
  d_test$selected <- bh$selected
  p_cutoff <- bh$cutoff
  
  # --- Evaluation ---
  selected_idx <- which(d_test$selected == TRUE)
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
    file = "p_value_misspecified.rds"
  )
}else{
  saveRDS(
    list(
      FDR = results$FDR,
      Power = results$Power
    ),
    file = "p_value_correct.rds"
  )
}

cat("--- Conformal Selection with p-values Results ---\n")
cat("We want to select Y <= ", c_threshold, ".\n")
cat("Alpha:", alpha, "\n")
cat("Replicate:", n_sim, "\n")
cat("Average Empirical FDR:", round(mean(results$FDR), 4), "\n")
cat("Average Empirical Power:", round(mean(results$Power), 4), "\n")