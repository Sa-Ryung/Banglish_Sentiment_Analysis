# Banglish Sentiment Analysis 🇧🇩

A comparative Machine Learning framework for sentiment classification of **Banglish** (Bengali written using the Roman script) text using classical NLP techniques, supervised learning models, and evaluation metrics in R.

---

## 📌 Project Overview

Banglish text presents unique NLP challenges due to informal spelling variations, mixed English-Bengali vocabulary, non-standard grammar, and excessive emoji usage. 

This project benchmarks and evaluates four classical Machine Learning approaches:
1. **Bag of Words (BoW) + Multinomial Naive Bayes**
2. **Bag of Words (BoW) + Logistic Regression (L1/L2 via `glmnet`)**
3. **TF-IDF + Gaussian Naive Bayes**
4. **TF-IDF + Logistic Regression (L1/L2 via `glmnet`)**

---

## 💾 Dataset

This project uses the **[Banglish Sentiment Dataset 2026](https://www.kaggle.com/datasets/mdsajjadullah/banglish-sentiment-dataset-2026)** hosted on Kaggle.

- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/mdsajjadullah/banglish-sentiment-dataset-2026)
- **Target Classes:** `negative`, `neutral`, `positive`
- **Format:** CSV file containing raw text comments and corresponding sentiment annotations.

---

## ✨ Features

- **Emoji Normalization:** Maps common emojis to contextual Banglish keywords (e.g., `😂` $\rightarrow$ `smile`, `😭` $\rightarrow$ `cry`).
- **Domain-Specific Preprocessing:** Removes a customized list of Banglish stopwords while retaining sentiment-bearing terms.
- **Sublinear TF-IDF Feature Scaling:** Uses $1 + \log(\text{TF})$ scaling with smoothed Inverse Document Frequency ($\text{IDF}$) calculated strictly from training data to avoid data leakage.
- **Stratified Splitting:** Uses `caret::createDataPartition` to ensure balanced 3-class target distributions (`negative`, `neutral`, `positive`).
- **Multi-Class Evaluation:** Computes Macro Precision, Macro Recall, and Macro F1-scores alongside raw Accuracy and Baseline comparisons.
- **Comprehensive Visualizations:** Generates sentiment distribution plots, text length distributions, grouped bar charts, line graphs, and radar charts.

---

## 🛠️ Requirements & Installation

Ensure you have **R (>= 4.0.0)** and **RStudio** installed.

### Required Packages

The script automatically detects and installs missing packages, but you can manually install them using:

```R
install.packages(c(
  "tm", "dplyr", "caTools", "stringr", "e1071", 
  "caret", "glmnet", "vioplot", "fmsb", "Matrix", "naivebayes"
))
<img width="784" height="1168" alt="Banglish_Sentiment_Analysis_Pipeline" src="https://github.com/user-attachments/assets/93247376-c52e-49e9-9eae-8a1329d11360" />
