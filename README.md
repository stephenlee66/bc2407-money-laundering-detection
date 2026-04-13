# 🏦 BC2407 — Money Laundering Detection

Individual assignment for BC2407 Analytics II: Advanced Predictive Techniques, focused on detecting money laundering cases using classification techniques.

## 📌 Project Overview

Individual assignment in developing a machine learning pipeline to detect money laundering across **1,054,321 banking transactions** spanning October 2022 to August 2023. Given severe class imbalance (only **0.94%** of transactions are laundering), the project prioritises **sensitivity** (correctly identifying true laundering cases) as missing a laundering transaction carries significantly higher risk than a false alarm.

The final Random Forest model achieved **86.33% sensitivity** and **96.10% specificity** with an overall accuracy of **96.01%**.

## 📂 Files

- [Question Paper](<CBA Question Paper.pdf>)
- [Assignment Submission](<Stephen Michael Lee Submission.pdf>)
- [R Script](<Stephen Michael Lee.R>)

## 📊 Dataset

| Variable | Type | Description |
|---|---|---|
| `Date` | Date | Transaction date (Oct 2022 - Aug 2023) |
| `Time` | Time | Transaction timestamp |
| `Sender_account` | Numeric | Sender account ID |
| `Receiver_account` | Numeric | Receiver account ID |
| `Amount` | Numeric | Transaction amount (Mixed currency, largely GBP) |
| `Payment_currency` | Categorical | Currency used for payment |
| `Received_currency` | Categorical | Currency received |
| `Sender_bank_location` | Categorical | Sender country |
| `Receiver_bank_location` | Categorical | Receiver country |
| `Payment_type` | Categorical | Payment method |
| `Is_laundering` | Binary | Target variable: 0 = Legitimate, 1 = Laundering |
| `Laundering_type` | Categorical | Type of laundering scheme (**dropped** later due to data leakage) |

## 🔍 Key EDA Findings

| # | Finding | Insight |
|---|---|---|
| 1 | Severe class imbalance, only **0.94%** laundering | High accuracy is not the best metric and to be addressed before modelling |
| 2 | **Cash Deposit (5.4%)** and **Cash Withdrawal (3.9%)** highest laundering rates | Harder to trace |
| 3 | Currency mismatch transactions **4.5× more likely** to involve laundering | 2.98% vs 0.67% same-currency rate |
| 4 | Laundering peaks during **business hours (8am–5pm)** | Mimics legitimate behaviour to avoid detection |
| 5 | Average daily laundering increased **~38%** from Nov 2022 to Aug 2023 | Growing threat requiring urgent automated detection |
| 6 | Laundering amounts show a **bimodal distribution** | Secondary cluster at ~100–200 consistent with structuring/smurfing |

## ⚙️ Methodology

### Pipeline Overview

```
Full Dataset (1,054,321)
        │
        ▼
  Stratified 70/30 Split
        │
        ├──► df.train.full (738,025)
        │         │
        │         ▼
        │   LOO Aggregated Feature Engineering
        │   (built from trainset only)
        │         │
        │         ▼
        │   Stratified Sample (100,000)
        │         │
        │         ▼
        │   ROSE Balancing (p=0.6, method="both")
        │   → df.train.balanced (100,000 | 59.76% laundering)
        │
        └──► df.test (316,296) — untouched full testset
```
> ⚠️ A stratified random split was chosen over a time-based split to ensure both the trainset and testset maintained a representative proportion of the rare laundering class (0.94%). With only 9,873 laundering cases across the full dataset, a time-based split risked concentrating laundering cases unevenly across train and test sets, potentially producing unreliable sensitivity estimates. However, this comes at the cost of temporal realism.

### Feature Engineering

**Basic Features:**

| Feature | Description |
|---|---|
| `Hour` | Hour of transaction extracted from Time |
| `Month` | Month of transaction extracted from Date |
| `Different_bank_location` | Whether sender and receiver banks are in different countries |
| `Currency_mismatch` | Whether payment and received currencies differ |

**Aggregated Features — Leave-One-Out (LOO):**

| Feature | Description | Why LOO? |
|---|---|---|
| `Sender_txn_count` | Number of other transactions by same sender | Excludes current row to prevent self-leakage |
| `Sender_avg_amount` | Average amount of other transactions by same sender | Excludes current row for unbiased estimate |
| `Receiver_txn_count` | Number of other transactions to same receiver | Flags high-frequency receiver accounts |
| `Pair_txn_count` | Number of other transactions between same sender-receiver pair | Detects repeated suspicious pairs |
| `Amount_vs_sender_avg` | Current amount divided by sender's average amount | Detects anomalous transaction amounts |

> ⚠️ All aggregated features were computed from the **trainset only** and applied to the testset separately, with missing values imputed using the global training median. This prevents data leakage.

### Variables Dropped

| Variable | Reason |
|---|---|
| `Sender_account`, `Receiver_account` | ID fields, used only for feature engineering, not model input |
| `Laundering_type` | **Data leakage**, only exists when laundering is already confirmed |
| `Date`, `Time` | Replaced by engineered `Hour` and `Month` |
| `Payment_currency`, `Received_currency` | Replaced by `Currency_mismatch` flag |
| `Sender_bank_location`, `Receiver_bank_location` | Replaced by `Different_bank_location` flag |
| `Different_bank_location` | **Extreme multicollinearity** in LR (VIF = 2,657), dropped from LR models only |

### Balancing Strategy

| Dataset | Observations | Laundering Cases | % Laundering |
|---|---|---|---|
| Full trainset before sampling | 738,025 | 6,911 | 0.94% |
| Sampled trainset before balancing | 100,000 | 936 | 0.94% |
| Sampled trainset after balancing | 100,000 | 59,764 | 59.76% |
| **Full testset** | **316,296** | **2,962** | **0.94%** |

The trainset was rebalanced using **ROSE** (`method = "both"`, `p = 0.6`), combining oversampling of the minority class and undersampling of the majority class. A proportion of **60% laundering** was chosen to bias the model slightly towards detecting the minority class, reflecting the higher cost of false negatives over false positives.

## 🤖 Models

### Logistic Regression

| Model | Description | Key Change |
|---|---|---|
| `lr.m1` | Full model — all variables | Baseline |
| `lr.m2` | Drop `Different_bank_location` | Resolves extreme multicollinearity (VIF = 2,647) |
| `lr.m3` | Stepwise AIC selection on `lr.m2` | **Final model — 8 X variables** |

### Random Forest

| Model | Configuration | Key Change |
|---|---|---|
| `rf.m1` | ntree=500, default cutoff | Baseline |
| `rf.m2` | ntree=500, cutoff=c(0.6, 0.4) | Lowers threshold to boost sensitivity |
| `rf.m3` | ntree=600, cutoff=c(0.6, 0.4) | More trees for stability — **Final model** |

A custom cutoff of `c(0.6, 0.4)` requires only **40% of tree votes** (instead of 50%) to classify a transaction as laundering, directly improving sensitivity at the cost of marginally higher false positives.

## 📈 Final Results

| Model | Complexity | Test Error (%) | Accuracy (%) | Sensitivity (%) | Specificity (%) |
|---|---|---|---|---|---|
| Logistic Regression (lr.m3) | 8 X variables | 30.89 | 69.11 | 74.24 | 69.06 |
| **Random Forest (rf.m3)** | **ntree=600, cutoff=(0.6,0.4)** | **3.99** | **96.01** | **86.33** | **96.10** |

### ✅ Recommended Model — Random Forest (rf.m3)

Random Forest significantly outperforms Logistic Regression across all metrics:
- **86.33% sensitivity** — correctly identifies ~86 in every 100 laundering transactions
- **96.10% specificity** — correctly clears ~96 in every 100 legitimate transactions  
- **3.99% test error** — evaluated on full 316,296 real-world transactions

### Variable Importance (RF — MeanDecreaseAccuracy)

| Rank | Variable | Insight |
|---|---|---|
| 1 | `Receiver_txn_count` | High-frequency receiver accounts are strong laundering signals |
| 2 | `Sender_txn_count` | High-frequency sender accounts indicate suspicious activity |
| 3 | `Amount_vs_sender_avg` | Transactions deviating from sender's norm are suspicious |
| 4 | `Payment_type` | Cash Deposit and Cash Withdrawal carry highest laundering risk |
| 5 | `Currency_mismatch` | Cross-currency transactions 4.5× more likely to be laundering |

## 🥲 Limitations

1. **Random train-test split instead of time-based split:** A random stratified split was used, meaning the trainset and testset contain transactions from the same time period (Oct 2022 – Aug 2023). In production deployment, a model would always predict future transactions based on past data. A time-based split would better reflect real-world conditions and provide a more conservative estimate of model performance.
2. **ROSE generates synthetic data:** The ROSE balancing method creates synthetic laundering transactions by interpolating between real ones. These synthetic cases may not represent the full diversity of real laundering behaviour, potentially causing the model to learn patterns that do not generalise to novel laundering schemes.
3. **LOO features assume account history is available:** The aggregated features (`Sender_txn_count`, `Receiver_txn_count` etc.) require prior transaction history for each account. For new accounts with no history, these features default to zero or median values, which may reduce detection accuracy for first-time laundering attempts

## 🛠️ Libraries Used

| Package | Purpose |
|---|---|
| `ggplot2` | Visualisation |
| `dplyr` | Data manipulation |
| `moments` | Skewness calculation |
| `lubridate` | Date/time handling |
| `caTools` | Stratified train-test split |
| `ROSE` | Class balancing |
| `randomForest` | Random Forest modelling |
| `car` | VIF multicollinearity analysis |
