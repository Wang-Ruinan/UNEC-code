## LOADING DATA
Sheridan_low <- readRDS("Sheridan_sim2.rds")
Sheridan_high <- readRDS("Sheridan_sim3.rds")

p_value_low <- readRDS("p_value_sim2.rds")
p_value_high <- readRDS("p_value_sim3.rds")

NP_low <- readRDS("NP_sim2.rds")
NP_high <- readRDS("NP_sim3.rds")

## AGGREGATE DATA
make_df <- function(res, method, setting) {
  rbind(
    data.frame(value = res$FDR, metric = "FDR"),
    data.frame(value = res$Power, metric = "Power")
  ) |>
    transform(method = method, setting = setting)
}

df_all <- rbind(
  make_df(Sheridan_low, "Sheridan Score", "Low-dimensional Model"),
  make_df(Sheridan_high, "Sheridan Score", "High-dimensional Model"),
  
  make_df(p_value_low, "p-value", "Low-dimensional Model"),
  make_df(p_value_high, "p-value", "High-dimensional Model"),
  
  make_df(NP_low, "NP Score", "Low-dimensional Model"),
  make_df(NP_high, "NP Score", "High-dimensional Model")
)
df_all$method <- factor(
  df_all$method,
  levels = c("Sheridan Score", "p-value", "NP Score")
)
#df_all$panel <- interaction(df_all$method, df_all$setting, df_all$metric, sep = " | ")

alpha <- 0.1

vlines <- aggregate(value ~ method + setting + metric,
                    data = df_all,
                    FUN = mean)

#vlines$label <- ifelse(vlines$metric == "FDR", "mean_FDR", "mean_Power")
vlines$type <- "mean"
vlines$color <- "blue"

# 加 alpha 线（只在 FDR）
vlines_alpha <- unique(vlines[vlines$metric == "FDR", c("method","setting","metric")])
vlines_alpha$value <- alpha
vlines_alpha$type <- "alpha"
vlines_alpha$color <- "red"

vlines <- rbind(vlines, vlines_alpha)

## PLOT
library(ggplot2)
library(patchwork)

## 分开 FDR 和 Power
df_fdr <- subset(df_all, metric == "FDR")
df_power <- subset(df_all, metric == "Power")

vlines_fdr <- subset(vlines, metric == "FDR" & type == "mean")
vlines_power <- subset(vlines, metric == "Power" & type == "mean")

## 左图：FDR
p_fdr <- ggplot(df_fdr, aes(x = value)) +
  
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "grey70",
                 color = NA,
                 alpha = 0.5) +
  
  geom_density(linewidth = 1) +
  
  geom_vline(data = vlines_fdr,
             aes(xintercept = value, color = "blue"),
             linetype = "dashed",
             linewidth = 0.8,
             show.legend = FALSE) +
  geom_vline(xintercept = alpha, color = "red", linetype = "dashed") +
  
  facet_grid(method ~ setting, scales = "free") +
  scale_color_identity() +
  
  labs(x = "FDR", y = "Density") +
  
  theme_bw() +
  theme(
    strip.text = element_text(size = 12),
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

## 右图：Power
p_power <- ggplot(df_power, aes(x = value)) +
  
  geom_histogram(aes(y = after_stat(density)),
                 bins = 30,
                 fill = "grey70",
                 color = NA,
                 alpha = 0.5) +
  
  geom_density(linewidth = 1) +
  
  geom_vline(data = vlines_power,
             aes(xintercept = value, color = "blue"),
             linetype = "dashed",
             linewidth = 0.8,
             show.legend = FALSE) +
  
  facet_grid(method ~ setting, scales = "free") +
  scale_color_identity() +
  
  labs(x = "Power", y = "Density") +
  
  theme_bw() +
  theme(
    strip.text = element_text(size = 12),
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

## 合成一个大图
fdr_tag <- p_fdr + labs(caption = "(a)") +
  theme(plot.caption = element_text(hjust = 0.5, size = 12))

power_tag <- p_power + labs(caption = "(b)") +
  theme(plot.caption = element_text(hjust = 0.5, size = 12))

p_all <- fdr_tag + power_tag + 
  plot_layout(ncol = 2, widths = c(1, 1)) +
  theme(legend.position = "bottom")
p_all

## 导出
pdf("Combined_Simulation_23_two_panels.pdf", width = 14, height = 8)
print(p_all)
dev.off()

## fdr&power
mean_summary <- aggregate(
  value ~ method + setting + metric,
  data = df_all,
  FUN = mean
)

mean_summary