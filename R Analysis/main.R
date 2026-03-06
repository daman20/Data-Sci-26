library(dplyr)

tourism <- read.csv("~/TSA SHIT/Real Data Sci 26/Data-Sci-26/API_ST.INT.ARVL_DS2_en_csv_v2_4621.csv", header=TRUE)
library(dplyr)
CTDC_global_synthetic_data_v2025 <- read.csv("~/TSA SHIT/Real Data Sci 26/Data-Sci-26/CTDC_global_synthetic_data_v2025.csv")
all_countries <- unique(CTDC_global_synthetic_data_v2025["CountryOfExploitation"])
all_citizen <- unique(CTDC_global_synthetic_data_v2025["citizenship"])



head(CTDC_global_synthetic_data_v2025)


all_countries <- unique(CTDC_global_synthetic_data_v2025["CountryOfExploitation"])
all_citizen <- unique(CTDC_global_synthetic_data_v2025["citizenship"])



library(tidyverse)

# Read data
df <- read_csv("~/TSA SHIT/Real Data Sci 26/Data-Sci-26/CTDC_global_synthetic_data_v2025.csv")
names(df) <- trimws(gsub("\uFEFF", "", names(df)))

# Pivot: rows = country, columns = year, values = incident count
pivot <- df %>%
  count(CountryOfExploitation, yearOfRegistration) %>%
  pivot_wider(
    names_from = yearOfRegistration,
    values_from = n,
    values_fill = 0
  ) %>%
  rename(Country = CountryOfExploitation)

# Sort columns chronologically (move Country to front)
year_cols <- sort(setdiff(names(pivot), "Country"))
pivot <- pivot %>% select(Country, all_of(year_cols))

# Add total column and sort by it descending
pivot <- pivot %>%
  mutate(Total = rowSums(across(where(is.numeric)))) %>%
  arrange(desc(Total))

# Add total row
total_row <- tibble(Country = "Total") %>%
  bind_cols(pivot %>% summarise(across(where(is.numeric), sum)))
pivot <- bind_rows(pivot, total_row)

traffic_summary <- pivot
remove(pivot)


# END DATA CLEANING

# export traffic_summary, tourism
write.csv(traffic_summary, "traffic.csv")
write.csv(tourism, "tourism.csv")


# ===========================================================
# Merge trafficking + tourism into a country-year panel
# ===========================================================
library(dplyr)

# Trafficking counts per country per year
trafficking_long <- df %>%
  count(CountryOfExploitation, yearOfRegistration) %>%
  rename(Country = CountryOfExploitation, Year = yearOfRegistration,
         trafficking_cases = n)

# Reshape tourism from wide to long
tourism_long <- tourism %>%
  select(Country.Code, starts_with("X")) %>%
  pivot_longer(cols = starts_with("X"),
               names_to = "Year",
               values_to = "tourist_arrivals") %>%
  mutate(Year = as.integer(gsub("X", "", Year))) %>%
  rename(Country = Country.Code) %>%
  filter(!is.na(tourist_arrivals))

# Merge on Country + Year
merged <- inner_join(trafficking_long, tourism_long,
                     by = c("Country", "Year"))

# Keep only years with reasonable overlap (2002-2019, drop 2020 COVID distortion)
merged <- merged %>% filter(Year >= 2002, Year <= 2019)

# Create output directory for figures
dir.create("figures", showWarnings = FALSE)

# ===========================================================
# 1. Scatter plot: trafficking cases vs tourist arrivals
# ===========================================================
library(ggplot2)

p1 <- ggplot(merged, aes(x = trafficking_cases, y = tourist_arrivals / 1e6)) +
  geom_point(alpha = 0.5, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title = "Relationship Between Trafficking Cases and Tourist Arrivals",
       x = "Number of Trafficking Cases",
       y = "Tourist Arrivals (millions)") +
  theme_minimal()
ggsave("figures/scatter_trafficking_vs_tourism.png", p1, width = 8, height = 6)

# ===========================================================
# 2. Log-log scatter (handles skew in both variables)
# ===========================================================
p2 <- ggplot(merged, aes(x = log(trafficking_cases + 1),
                          y = log(tourist_arrivals + 1))) +
  geom_point(alpha = 0.5, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title = "Log-Log: Trafficking Cases vs Tourist Arrivals",
       x = "Log(Trafficking Cases + 1)",
       y = "Log(Tourist Arrivals + 1)") +
  theme_minimal()
ggsave("figures/scatter_log_trafficking_vs_tourism.png", p2, width = 8, height = 6)

# ===========================================================
# 3. Year-over-year change analysis
#    Does an INCREASE in trafficking predict a DECREASE in tourism?
# ===========================================================
merged_changes <- merged %>%
  arrange(Country, Year) %>%
  group_by(Country) %>%
  mutate(
    trafficking_change = trafficking_cases - lag(trafficking_cases),
    tourism_change_pct = (tourist_arrivals - lag(tourist_arrivals)) / lag(tourist_arrivals) * 100
  ) %>%
  filter(!is.na(trafficking_change), !is.na(tourism_change_pct))

p3 <- ggplot(merged_changes, aes(x = trafficking_change, y = tourism_change_pct)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  labs(title = "Change in Trafficking vs Change in Tourism (Year-over-Year)",
       x = "Change in Trafficking Cases",
       y = "Change in Tourist Arrivals (%)") +
  theme_minimal()
ggsave("figures/scatter_yoy_changes.png", p3, width = 8, height = 6)

# ===========================================================
# 4. Aggregated time series: global totals per year
# ===========================================================
global_ts <- merged %>%
  group_by(Year) %>%
  summarise(total_trafficking = sum(trafficking_cases),
            total_tourism = sum(tourist_arrivals) / 1e6,
            .groups = "drop")

p4 <- ggplot(global_ts, aes(x = Year)) +
  geom_line(aes(y = total_trafficking, color = "Trafficking Cases"), linewidth = 1) +
  geom_point(aes(y = total_trafficking, color = "Trafficking Cases")) +
  scale_color_manual(values = c("Trafficking Cases" = "red")) +
  labs(title = "Global Trafficking Cases Over Time",
       x = "Year", y = "Total Trafficking Cases", color = "") +
  theme_minimal()

p5 <- ggplot(global_ts, aes(x = Year)) +
  geom_line(aes(y = total_tourism, color = "Tourist Arrivals (M)"), linewidth = 1) +
  geom_point(aes(y = total_tourism, color = "Tourist Arrivals (M)")) +
  scale_color_manual(values = c("Tourist Arrivals (M)" = "steelblue")) +
  labs(title = "Global Tourist Arrivals Over Time (Matched Countries)",
       x = "Year", y = "Tourist Arrivals (millions)", color = "") +
  theme_minimal()

ggsave("figures/timeseries_trafficking.png", p4, width = 8, height = 5)
ggsave("figures/timeseries_tourism.png", p5, width = 8, height = 5)

# ===========================================================
# 5. Top-10 country comparison: high trafficking vs tourism
# ===========================================================
top_trafficked <- merged %>%
  group_by(Country) %>%
  summarise(total_trafficking = sum(trafficking_cases),
            avg_tourism = mean(tourist_arrivals) / 1e6,
            .groups = "drop") %>%
  arrange(desc(total_trafficking)) %>%
  slice_head(n = 10)

p6 <- ggplot(top_trafficked, aes(x = reorder(Country, -total_trafficking))) +
  geom_col(aes(y = total_trafficking), fill = "tomato", alpha = 0.8) +
  labs(title = "Top 10 Countries by Trafficking Cases",
       x = "Country", y = "Total Trafficking Cases") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("figures/bar_top10_trafficking.png", p6, width = 8, height = 5)

p7 <- ggplot(top_trafficked, aes(x = reorder(Country, -total_trafficking),
                                  y = avg_tourism)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  labs(title = "Avg Tourist Arrivals for Top 10 Trafficking Countries",
       x = "Country", y = "Avg Tourist Arrivals (millions)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("figures/bar_top10_tourism.png", p7, width = 8, height = 5)

# ===========================================================
# 6. Statistical tests
# ===========================================================

cat("\n====== STATISTICAL ANALYSIS ======\n\n")

# Pearson correlation
cor_test <- cor.test(merged$trafficking_cases, merged$tourist_arrivals)
cat("--- Pearson Correlation (raw) ---\n")
cat(sprintf("  r = %.4f, p = %.4e\n\n", cor_test$estimate, cor_test$p.value))

# Pearson correlation on logs
cor_log <- cor.test(log(merged$trafficking_cases + 1),
                    log(merged$tourist_arrivals + 1))
cat("--- Pearson Correlation (log-transformed) ---\n")
cat(sprintf("  r = %.4f, p = %.4e\n\n", cor_log$estimate, cor_log$p.value))

# Simple linear regression
model1 <- lm(tourist_arrivals ~ trafficking_cases, data = merged)
cat("--- Linear Regression: tourism ~ trafficking ---\n")
print(summary(model1))

# Regression with country and year fixed effects
model2 <- lm(tourist_arrivals ~ trafficking_cases + factor(Country) + factor(Year),
             data = merged)
cat("\n--- Fixed Effects Regression: tourism ~ trafficking + country FE + year FE ---\n")
cat(sprintf("  Trafficking coefficient: %.2f\n", coef(model2)["trafficking_cases"]))
cat(sprintf("  p-value: %.4e\n", summary(model2)$coefficients["trafficking_cases", 4]))
cat(sprintf("  Adjusted R-squared: %.4f\n\n", summary(model2)$adj.r.squared))

# Year-over-year changes regression
model3 <- lm(tourism_change_pct ~ trafficking_change, data = merged_changes)
cat("--- YoY Changes Regression: tourism_change% ~ trafficking_change ---\n")
print(summary(model3))

# Spearman rank correlation (robust to outliers)
spearman <- cor.test(merged$trafficking_cases, merged$tourist_arrivals,
                     method = "spearman")
cat("\n--- Spearman Rank Correlation ---\n")
cat(sprintf("  rho = %.4f, p = %.4e\n", spearman$estimate, spearman$p.value))

# Granger-like lagged analysis: does last year's trafficking predict this year's tourism?
merged_lag <- merged %>%
  arrange(Country, Year) %>%
  group_by(Country) %>%
  mutate(lag_trafficking = lag(trafficking_cases)) %>%
  filter(!is.na(lag_trafficking))

model_lag <- lm(tourist_arrivals ~ lag_trafficking + factor(Country) + factor(Year),
                data = merged_lag)
cat("\n--- Lagged Regression: tourism ~ lag(trafficking) + country FE + year FE ---\n")
cat(sprintf("  Lag trafficking coefficient: %.2f\n",
            coef(model_lag)["lag_trafficking"]))
cat(sprintf("  p-value: %.4e\n",
            summary(model_lag)$coefficients["lag_trafficking", 4]))

cat("\n====== END OF ANALYSIS ======\n")
