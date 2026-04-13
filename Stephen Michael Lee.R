# ============================================================================
# IMPORT LIBRARIES
# ============================================================================

library(ggplot2) # Visuals
library(dplyr) # Easy data manipulation
library(moments) # Check skewness
library(lubridate) # Easy datetime handling
library(caTools) # Train-test split
library(ROSE) # Re-balancing
library(randomForest) # Random forest
library(car) # VIF

# ============================================================================
# LOAD RAW DATA & BASIC CHECKS
# ============================================================================

df.raw <- read.csv("data2.csv", header = TRUE, na.strings = c("NA", "na", "N/A", "", ".", "m", "M"))

dim(df.raw)
str(df.raw)
head(df.raw, 5)
summary(df.raw)

# Check missing & duplicates
sum(is.na(df.raw))
sum(duplicated(df.raw))

# Check data date range
min(df.raw$Date); max(df.raw$Date)
length(unique(df.raw$Date))

# Check skewness
skewness(df.raw$Amount)
skewness(df.raw$Amount[df.raw$Is_laundering == 0])
skewness(df.raw$Amount[df.raw$Is_laundering == 1])

unique(df.raw$Payment_type)
unique(df.raw$Sender_bank_location)
unique(df.raw$Receiver_bank_location)
unique(df.raw$Payment_currency)
unique(df.raw$Received_currency)

# ============================================================================
# EDA
# ============================================================================

# 1. Class balance
class_balance_tbl <- df.raw %>%
  group_by(Is_laundering) %>%
  summarise(Count = n()) %>%
  mutate(Proportion = round(Count / sum(Count) * 100, 2),
         Label = ifelse(Is_laundering == 1, "Laundering", "Legitimate"))

ggplot(class_balance_tbl, aes(x = Label, y = Count, fill = Label)) +
  geom_col(width = 0.5, alpha = 0.9) +
  geom_text(aes(label = paste0(formatC(Count, format = "d", big.mark = ","),
                               "\n(", Proportion, "%)")),
            vjust = -0.3, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = c("Laundering" = "firebrick", "Legitimate" = "steelblue")) +
  scale_y_continuous(labels = function(x) formatC(x, format = "d", big.mark = ","),
                     expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Class Balance Distribution") +
  theme(legend.position = "none")

# 2. Laundering rate by payment type
payment_tbl <- df.raw %>%
  group_by(Payment_type, Is_laundering) %>%
  summarise(Count = n(), .groups = "drop") %>%
  group_by(Payment_type) %>%
  mutate(Pct_laundering = round(Count / sum(Count) * 100, 2)) %>%
  filter(Is_laundering == 1) %>%
  arrange(desc(Pct_laundering))

ggplot(payment_tbl, aes(x = reorder(Payment_type, Pct_laundering), y = Pct_laundering)) +
  geom_col(fill = "firebrick", alpha = 0.85, width = 0.6) +
  geom_text(aes(label = paste0(Pct_laundering, "%")),
            hjust = -0.2, size = 5, fontface = "bold") +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Laundering Rate by Payment Type",
       x = "Payment Type", y = "Laundering Rate (%)")

# 3. Count of laundering transactions per payment type
ggplot(payment_tbl, aes(x = reorder(Payment_type, Count), y = Count)) +
  geom_col(fill = "firebrick", alpha = 0.85, width = 0.6) +
  geom_text(aes(label = formatC(Count, format = "d", big.mark = ",")),
            hjust = -0.2, size = 3.5, fontface = "bold") +
  coord_flip() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Number of Laundering Transactions by Payment Type",
       x = "Payment Type", y = "Number of Laundering Transactions")

# 4. Amount distributions
ggplot(df.raw %>% filter(Is_laundering == 1), aes(x = Amount)) +
  geom_histogram(bins = 60, fill = "firebrick", alpha = 0.85) +
  scale_x_log10(labels = function(x) formatC(x, format = "d", big.mark = ",")) +
  labs(title = "Amount Distribution (Laundering Transactions)",
       x = "Amount (Log Scale)", y = "Count")

ggplot(df.raw %>% filter(Is_laundering == 0), aes(x = Amount)) +
  geom_histogram(bins = 60, fill = "steelblue", alpha = 0.85) +
  scale_x_log10(labels = function(x) formatC(x, format = "d", big.mark = ",")) +
  labs(title = "Amount Distribution (Legitimate Transactions)",
       x = "Amount (Log Scale)", y = "Count")

# 5. Time series of laundering by date
laundering_ts <- df.raw %>%
  filter(Is_laundering == 1) %>%
  group_by(Date) %>%
  summarise(Count = n()) %>%
  mutate(Date = as.Date(Date))

ggplot(laundering_ts, aes(x = Date, y = Count, group = 1)) +
  geom_line(color = "firebrick", alpha = 0.8) +
  geom_smooth(method = "loess", color = "darkred", se = FALSE, linetype = "dashed") +
  scale_x_date(date_breaks = "2 months", date_labels = "%b %Y") +
  labs(title = "Daily Laundering Transactions Over Time",
       subtitle = "Oct 2022 - Aug 2023", x = "Date",
       y = "Number of Laundering Transactions")

# 6. Average daily laundering per month
laundering_monthly <- df.raw %>%
  filter(Is_laundering == 1) %>%
  mutate(Month_Year = format(as.Date(Date), "%Y-%m")) %>%
  group_by(Month_Year) %>%
  summarise(Avg_Daily = mean(table(Date))) %>%
  mutate(Month_Year = as.Date(paste0(Month_Year, "-01")))

ggplot(laundering_monthly, aes(x = Month_Year, y = Avg_Daily, group = 1)) +
  geom_line(color = "firebrick", alpha = 0.8) +
  geom_point(color = "firebrick", size = 2.5) +
  scale_x_date(date_breaks = "1 month", date_labels = "%b %Y") +
  labs(title = "Average Daily Laundering Transactions by Month",
       subtitle = "Oct 2022 - Aug 2023", x = "Month",
       y = "Avg Daily Laundering Transactions")

# 7. Laundering by hour
hour_tbl <- df.raw %>%
  filter(Is_laundering == 1) %>%
  mutate(Hour = hour(hms::as_hms(Time))) %>%
  group_by(Hour) %>%
  summarise(Count = n())

ggplot(hour_tbl, aes(x = Hour, y = Count)) +
  geom_col(fill = "firebrick", alpha = 0.85) +
  labs(title = "Laundering Transactions by Hour of Day",
       x = "Hour", y = "Count")

# 8. Laundering by day of week
dow_tbl <- df.raw %>%
  filter(Is_laundering == 1) %>%
  mutate(DOW = wday(as.Date(Date), label = TRUE, abbr = FALSE)) %>%
  group_by(DOW) %>%
  summarise(Count = n())

ggplot(dow_tbl, aes(x = DOW, y = Count)) +
  geom_col(fill = "firebrick", alpha = 0.85) +
  labs(title = "Laundering Transactions by Day of Week",
       x = "Day", y = "Count")

# 9. Laundering by sender/receiver country
sender_tbl <- df.raw %>%
  filter(Is_laundering == 1) %>%
  group_by(Sender_bank_location) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count))

ggplot(sender_tbl, aes(x = reorder(Sender_bank_location, Count), y = Count)) +
  geom_col(fill = "firebrick", alpha = 0.85) +
  geom_text(aes(label = Count), hjust = -0.2, size = 3.5) +
  coord_flip() +
  labs(title = "Laundering Transactions by Sender Country",
       x = "Country", y = "Count")

receiver_tbl <- df.raw %>%
  filter(Is_laundering == 1) %>%
  group_by(Receiver_bank_location) %>%
  summarise(Count = n()) %>%
  arrange(desc(Count))

ggplot(receiver_tbl, aes(x = reorder(Receiver_bank_location, Count), y = Count)) +
  geom_col(fill = "firebrick", alpha = 0.85) +
  geom_text(aes(label = Count), hjust = -0.2, size = 3.5) +
  coord_flip() +
  labs(title = "Laundering Transactions by Receiver Country",
       x = "Country", y = "Count")

# 10. Currency mismatch vs laundering rate
mismatch_tbl <- df.raw %>%
  mutate(Currency_mismatch = ifelse(Payment_currency != Received_currency,
                                    "Mismatch", "Same")) %>%
  group_by(Currency_mismatch, Is_laundering) %>%
  summarise(Count = n(), .groups = "drop") %>%
  group_by(Currency_mismatch) %>%
  mutate(Pct = round(Count / sum(Count) * 100, 2)) %>%
  filter(Is_laundering == 1)

ggplot(mismatch_tbl, aes(x = Currency_mismatch, y = Pct, fill = Currency_mismatch)) +
  geom_col(width = 0.5, alpha = 0.85) +
  geom_text(aes(label = paste0(Pct, "%")),
            vjust = -0.4, size = 6, fontface = "bold") +
  scale_fill_manual(values = c("Mismatch" = "firebrick", "Same" = "steelblue")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Laundering Rate by Currency Mismatch",
       x = "Currency Type", y = "Laundering Rate (%)") +
  theme(legend.position = "none")

# ============================================================================
# INITIAL DATA PREPARATION
# ============================================================================

# Convert date & time
df.raw$Date <- as.Date(df.raw$Date, format = "%Y-%m-%d")
df.raw$Time <- hms::as_hms(df.raw$Time)

# Basic feature engineering
df.clean <- df.raw %>%
  mutate(
    Hour = hour(Time),
    Month = month(Date),
    Is_laundering = factor(Is_laundering, levels = c(0, 1), labels = c("Legitimate", "Laundering")),
    Different_bank_location = ifelse(Sender_bank_location != Receiver_bank_location, 1, 0),
    Currency_mismatch = ifelse(Payment_currency != Received_currency, 1, 0)
  )

# Initial feature selection
drop_cols <- c("Time", "Date", "Laundering_type", "Payment_currency",
               "Received_currency", "Sender_bank_location", "Receiver_bank_location")

# Categorical encoding
df.model <- df.clean %>%
  select(-any_of(drop_cols)) %>%
  mutate(
    Payment_type = factor(Payment_type),
    Hour = as.numeric(Hour),
    Month = as.numeric(Month),
    Different_bank_location = factor(Different_bank_location),
    Currency_mismatch = factor(Currency_mismatch)
  )

# ============================================================================
# 70/30 TRAIN-TEST SPLIT
# ============================================================================

set.seed(2026)
split <- sample.split(df.model$Is_laundering, SplitRatio = 0.7)
df.train.full <- df.model[split == TRUE, ]
df.test <- df.model[split == FALSE, ]

# ============================================================================
# AGGREGATED FEATURE ENGINEERING
# ============================================================================

# Leave-One-Out sender stats
sender_loo <- df.train.full %>%
  group_by(Sender_account) %>%
  mutate(
    Sender_txn_count  = n() - 1,
    Sender_avg_amount = ifelse(
      (n() - 1) > 0,
      (sum(Amount) - Amount) / (n() - 1),
      NA_real_
    )
  ) %>%
  ungroup()

# Fill NAs (senders with a single transaction) with the global training median
global_median_amount <- median(df.train.full$Amount, na.rm = TRUE)
sender_loo$Sender_avg_amount[is.na(sender_loo$Sender_avg_amount)] <- global_median_amount

# Leave-One-Out receiver stats
receiver_loo <- df.train.full %>%
  group_by(Receiver_account) %>%
  mutate(Receiver_txn_count = n() - 1) %>%
  ungroup()

# Leave-One-Out pair stats
pair_loo <- df.train.full %>%
  group_by(Sender_account, Receiver_account) %>%
  mutate(Pair_txn_count = n() - 1) %>%
  ungroup()

# Attach LOO features to train set
df.train.full <- df.train.full %>%
  mutate(
    Sender_txn_count = sender_loo$Sender_txn_count,
    Sender_avg_amount = sender_loo$Sender_avg_amount,
    Receiver_txn_count = receiver_loo$Receiver_txn_count,
    Pair_txn_count = pair_loo$Pair_txn_count,
    Amount_vs_sender_avg = Amount / pmax(Sender_avg_amount, 1)
  )

# Aggregate from train
sender_stats <- df.train.full %>%
  group_by(Sender_account) %>%
  summarise(Sender_txn_count = n(), Sender_avg_amount = mean(Amount, na.rm = TRUE), .groups = "drop")

receiver_stats <- df.train.full %>%
  group_by(Receiver_account) %>%
  summarise(Receiver_txn_count = n(), .groups = "drop")

pair_stats <- df.train.full %>%
  group_by(Sender_account, Receiver_account) %>%
  summarise(Pair_txn_count = n(), .groups = "drop")

# Apply on test
median_sender_avg <- median(sender_stats$Sender_avg_amount, na.rm = TRUE)
df.test <- df.test %>%
  left_join(sender_stats, by = "Sender_account") %>%
  left_join(receiver_stats, by = "Receiver_account") %>%
  left_join(pair_stats, by = c("Sender_account", "Receiver_account")) %>%
  mutate(
    Sender_txn_count = ifelse(is.na(Sender_txn_count), 0, Sender_txn_count),
    Receiver_txn_count = ifelse(is.na(Receiver_txn_count), 0, Receiver_txn_count),
    Pair_txn_count = ifelse(is.na(Pair_txn_count), 0, Pair_txn_count),
    Sender_avg_amount = ifelse(is.na(Sender_avg_amount),  median_sender_avg, Sender_avg_amount),
    Amount_vs_sender_avg = Amount / pmax(Sender_avg_amount, 1)
  )

# ============================================================================
# SAMPLE TRAIN SET FOR SPEED
# ============================================================================

set.seed(2026)
n_sample   <- 100000
laundering <- df.train.full[df.train.full$Is_laundering == "Laundering", ]
legitimate <- df.train.full[df.train.full$Is_laundering == "Legitimate",  ]
prop_l     <- nrow(laundering) / nrow(df.train.full)
n_l        <- round(n_sample * prop_l)
n_leg      <- n_sample - n_l

df.train <- rbind(laundering[sample(nrow(laundering), n_l),  ], legitimate[sample(nrow(legitimate), n_leg), ])
df.train <- df.train[sample(nrow(df.train)), ]

# Remove account IDs (identifiers, not predictors)
df.train <- df.train %>% select(-Sender_account, -Receiver_account)
df.test  <- df.test  %>% select(-Sender_account, -Receiver_account)

# ============================================================================
# BALANCE TRAIN SET
# ============================================================================

df.train.balanced <- ovun.sample(Is_laundering ~ ., data = df.train,
                                 N = nrow(df.train), p = 0.6, method = "both", seed = 2026)$data

# Verify proportions
prop.table(table(df.train.balanced$Is_laundering))
prop.table(table(df.test$Is_laundering))

# Summary report
report_tbl <- data.frame(
  Dataset = c(
    "Full trainset before sampling",
    "Sampled trainset before balancing",
    "Sampled trainset after balancing",
    "Full Testset"
  ),
  Number_observations = c(
    nrow(df.train.full), nrow(df.train),
    nrow(df.train.balanced), nrow(df.test)
  ),
  Number_laundering = c(
    sum(df.train.full$Is_laundering == "Laundering"),
    sum(df.train$Is_laundering == "Laundering"),
    sum(df.train.balanced$Is_laundering == "Laundering"),
    sum(df.test$Is_laundering == "Laundering")
  ),
  Percentage_laundering = c(
    round(mean(df.train.full$Is_laundering == "Laundering") * 100, 2),
    round(mean(df.train$Is_laundering == "Laundering") * 100, 2),
    round(mean(df.train.balanced$Is_laundering == "Laundering") * 100, 2),
    round(mean(df.test$Is_laundering == "Laundering") * 100, 2)
  )
)

report_tbl

# ============================================================================
# MODELLING
# ============================================================================

# ----------------------------------------------------------------------------
# Logistic Regression
# ----------------------------------------------------------------------------

# Model 1: Full model baseline
lr.m1 <- glm(Is_laundering ~ ., data = df.train.balanced, family = binomial)
summary(lr.m1)
vif(lr.m1)

lr.m1.prob <- predict(lr.m1, df.test, type = "response")
lr.m1.pred <- factor(ifelse(lr.m1.prob > 0.5, "Laundering", "Legitimate"), 
                     levels = levels(df.test$Is_laundering))
lr.m1.cm <- table(Actual = df.test$Is_laundering, Predicted = lr.m1.pred)
lr.m1.cm

# Model 2: Drop collinear variables 
lr.m2 <- glm(Is_laundering ~ . - Different_bank_location, 
             data = df.train.balanced, family = binomial)
summary(lr.m2)
vif(lr.m2)

lr.m2.prob <- predict(lr.m2, df.test, type = "response")
lr.m2.pred <- factor(ifelse(lr.m2.prob > 0.5, "Laundering", "Legitimate"), 
                     levels = levels(df.test$Is_laundering))
lr.m2.cm <- table(Actual = df.test$Is_laundering, Predicted = lr.m2.pred)
lr.m2.cm

# Model 3: Step wise AIC selection
lr.m3 <- step(lr.m2, direction = "both", trace = FALSE)
summary(lr.m3)
vif(lr.m3)

lr.m3.prob <- predict(lr.m3, df.test, type = "response")
lr.m3.pred <- factor(ifelse(lr.m3.prob > 0.5, "Laundering", "Legitimate"), 
                     levels = levels(df.test$Is_laundering))
lr.m3.cm <- table(Actual = df.test$Is_laundering, Predicted = lr.m3.pred)
lr.m3.cm

# ----------------------------------------------------------------------------
# Random Forest
# ----------------------------------------------------------------------------

# Model 1: Baseline
rf.m1 <- randomForest(Is_laundering ~ ., data = df.train.balanced,
                      na.action = na.omit, importance = TRUE)
rf.m1.pred <- predict(rf.m1, df.test)
rf.m1.cm   <- table(Actual = df.test$Is_laundering, Predicted = rf.m1.pred)
rf.m1.cm

# Model 2: Adjusted cutoff to boost sensitivity
rf.m2 <- randomForest(Is_laundering ~ ., data = df.train.balanced,
                      cutoff = c(0.6, 0.4), na.action = na.omit, importance = TRUE)
rf.m2.pred <- predict(rf.m2, df.test)
rf.m2.cm   <- table(Actual = df.test$Is_laundering, Predicted = rf.m2.pred)
rf.m2.cm

# Model 3: More trees
rf.m3 <- randomForest(Is_laundering ~ ., data = df.train.balanced,
                      cutoff = c(0.6, 0.4), ntree = 600, na.action = na.omit, importance = TRUE)
rf.m3.pred <- predict(rf.m3, df.test)
rf.m3.cm   <- table(Actual = df.test$Is_laundering, Predicted = rf.m3.pred)
rf.m3.cm

# ============================================================================
# MODEL EVALUATION METRICS
# ============================================================================

compute_metrics <- function(cm, model_name, complexity) {
  TN <- cm["Legitimate", "Legitimate"]
  FP <- cm["Legitimate", "Laundering"]
  FN <- cm["Laundering", "Legitimate"]
  TP <- cm["Laundering", "Laundering"]

  accuracy <- (TP + TN) / sum(cm)
  error <- 1 - accuracy
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)

  data.frame(
    Model = model_name,
    Complexity = complexity,
    Accuracy = round(accuracy * 100, 2),
    Test_Error = round(error * 100, 2),
    Sensitivity = round(sensitivity * 100, 2),
    Specificity = round(specificity * 100, 2)
  )
}

results_tbl <- rbind(
  compute_metrics(lr.m1.cm, "Logistic Regression (lr.m1)",
                  "11 X variables"),
  compute_metrics(lr.m2.cm, "Logistic Regression (lr.m2)",
                  "10 X variables"),
  compute_metrics(lr.m3.cm, "Logistic Regression (lr.m3)",
                  "8 X variables"),
  compute_metrics(rf.m1.cm, "Random Forest (rf.m1)",
                  "500 trees, default cutoff"),
  compute_metrics(rf.m2.cm, "Random Forest (rf.m2)",
                  "500 trees, cutoff=(0.6,0.4)"),
  compute_metrics(rf.m3.cm, "Random Forest (rf.m3)",
                  "600 trees, cutoff=(0.6,0.4)")
)

results_tbl

# ============================================================================
# VARIABLE IMPORTANCE (Best RF model)
# ============================================================================

importance(rf.m3)
varImpPlot(rf.m3, type = 1, main = "Variable Importance (MeanDecreaseAccuracy)")
