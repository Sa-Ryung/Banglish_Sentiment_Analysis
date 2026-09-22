# =========================================================
# BANGLISH SENTIMENT ANALYSIS
# BoW + Multinomial Naive Bayes
# BoW + Logistic Regression
# TF-IDF + Gaussian Naive Bayes
# TF-IDF + Logistic Regression
# =========================================================


# ---------------------------------------------------------
# Packages Installation & Loading
# ---------------------------------------------------------

required_packages <- c(
  "tm", "dplyr", "caTools", "stringr", "e1071",
  "caret", "glmnet", "vioplot", "fmsb", "Matrix", "naivebayes"
)

new_packages <- required_packages[
  !(required_packages %in% rownames(installed.packages()))
]

if (length(new_packages) > 0) {
  install.packages(new_packages)
}

library(tm)
library(dplyr)
library(caTools)
library(stringr)
library(e1071)
library(caret)
library(glmnet)
library(vioplot)
library(fmsb)
library(Matrix)
library(naivebayes)


# ---------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------

data_path <- Sys.getenv(
  "BANGLISH_DATA_PATH",
  unset = "D:/10th_sem/IDS/FPROJECT/Banglish_Dataset.csv"
)


# ---------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------

data <- read.csv(
  data_path,
  encoding = "UTF-8",
  stringsAsFactors = FALSE
)

data$sentiment <- factor(
  data$sentiment,
  levels = c("negative", "neutral", "positive")
)


# ---------------------------------------------------------
# 2. Emoji Replacement
# ---------------------------------------------------------

custom_emojis <- c(
  "😂" = " smile ",  "🙄" = " annoyed ",  "😅" = " nervous ",
  "😭" = " cry ",    "😞" = " sad ",      "😡" = " angry ",
  "👎" = " bad ",    "❌" = " wrong ",    "💔" = " heartbreak ",
  "🤮" = " disgust ", "🤬" = " angry ",   "😍" = " love ",
  "🔥" = " fire ",   "🥳" = " celebrating ", "🙌" = " celebrate ",
  "💥" = " explosion ", "🥰" = " love ", "✨" = " sparkles ",
  "✅" = " correct ", "❤" = " love "
)

data$text <- str_replace_all(data$text, custom_emojis)
data$text <- str_squish(data$text)


# ---------------------------------------------------------
# 3. Banglish Stopwords
# ---------------------------------------------------------

banglish_stopwords <- c(
  "eta", "to", "ta", "er", "ra", "te", "e", "ti", "tar", "tai", "asolei",
  "mone", "hoy", "sotti", "ekdom", "puro", "khub", "onekta", "ki",
  "bolbo", "ar", "bas", "vibe", "shunlam", "dekhi", "vabi", "amar",
  "ajker", "ei", "oir", "boss", "brother", "friend", "family", "class",
  "exam", "job", "traffic", "phone", "khawa", "weather", "internet",
  "eid", "sokal", "bazar", "bus", "coding", "match", "laptop", "review",
  "assignment", "chayer", "adda", "gaan", "movie", "result", "ache",
  "ase", "chilo", "hoilo", "hobe", "jani", "janina", "parlam", "parbo",
  "laglo", "lagse", "choltese", "thaake", "asol", "mathay", "kora",
  "kintu", "tobe", "jodi", "tahole", "ebong", "ba", "hoyto", "amra",
  "tumi", "tui", "apni", "tara", "tomar", "tar", "ora", "amake",
  "tomake", "tare", "mama", "je", "hae", "hmm"
)


# ---------------------------------------------------------
# 4. Text Preprocessing
# ---------------------------------------------------------

corpus <- VCorpus(VectorSource(data$text))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, banglish_stopwords)
corpus <- tm_map(corpus, stripWhitespace)

data$text <- sapply(corpus, function(x) x$content)
data$text <- str_squish(data$text)


# ---------------------------------------------------------
# 5. Remove Empty Texts
# ---------------------------------------------------------

valid_text <- !is.na(data$text) & str_squish(data$text) != ""
data_clean <- data[valid_text, ]

data_clean$sentiment <- factor(
  data_clean$sentiment,
  levels = c("negative", "neutral", "positive")
)

data_clean$words <- str_count(str_squish(data_clean$text), "\\S+")
corpus_clean <- VCorpus(VectorSource(data_clean$text))


# ---------------------------------------------------------
# 6. Exploratory Data Visualizations
# ---------------------------------------------------------

par(mfrow = c(1, 3))

sentiment_counts <- table(data_clean$sentiment)
barplot(
  sentiment_counts,
  main = "Sentiment Distribution",
  col = c("#E41A1C", "#377EB8", "#4DAF4A"),
  xlab = "Sentiment Class",
  ylab = "Number of Comments"
)

mean_words <- mean(data_clean$words)
median_words <- median(data_clean$words)

hist(
  data_clean$words,
  main = "Distribution of Text Length",
  xlab = "Words per text",
  col = "lightblue", border = "white", breaks = 15
)
abline(v = mean_words, col = "red", lwd = 2)
abline(v = median_words, col = "darkgreen", lwd = 2, lty = 2)
legend("topright", legend = c("Mean", "Median"),
       col = c("red", "darkgreen"), lty = c(1, 2), lwd = 2)

vioplot(
  words ~ sentiment,
  data = data_clean,
  main = "Text Length by Sentiment",
  col = c("#80FF80", "#FF8080", "#80B3FF"),
  xlab = "Sentiment", ylab = "Words per text"
)

par(mfrow = c(1, 1))


# ---------------------------------------------------------
# 7. Stratified Train/Test Split
# ---------------------------------------------------------
# FIX: use createDataPartition (caret) instead of sample.split
# sample.split is designed for binary targets and does not
# guarantee class balance on a 3-class factor.

set.seed(123)

train_idx <- createDataPartition(
  data_clean$sentiment,
  p = 0.80,
  list = FALSE
)

train_corpus <- corpus_clean[train_idx]
test_corpus  <- corpus_clean[-train_idx]

y_train <- data_clean$sentiment[train_idx]
y_test  <- data_clean$sentiment[-train_idx]

cat("\n=========================================\n")
cat("          DATA SPLIT INFORMATION\n")
cat("=========================================\n")
cat("Training samples :", length(y_train), "\n")
cat("Testing samples  :", length(y_test), "\n")

cat("\nClass distribution (train):\n")
print(table(y_train) / length(y_train))
cat("\nClass distribution (test):\n")
print(table(y_test) / length(y_test))


# ---------------------------------------------------------
# 8. Macro Metrics Function
# ---------------------------------------------------------

calc_macro_metrics <- function(cm_result) {
  precision_vec <- cm_result$byClass[, "Pos Pred Value"]
  recall_vec <- cm_result$byClass[, "Sensitivity"]
  
  precision_vec[is.na(precision_vec)] <- 0
  recall_vec[is.na(recall_vec)] <- 0
  
  precision <- mean(precision_vec)
  recall <- mean(recall_vec)
  
  f1_vec <- ifelse(
    (precision_vec + recall_vec) == 0,
    0,
    2 * (precision_vec * recall_vec) / (precision_vec + recall_vec)
  )
  
  f1 <- mean(f1_vec)
  
  return(c(precision = precision, recall = recall, f1 = f1))
}


# =========================================================
# 9. BAG OF WORDS
# =========================================================

dtm_train_bow <- DocumentTermMatrix(train_corpus)
dtm_train_bow_clean <- removeSparseTerms(dtm_train_bow, 0.99)
train_vocab <- Terms(dtm_train_bow_clean)

dtm_test_bow <- DocumentTermMatrix(
  test_corpus,
  control = list(dictionary = train_vocab)
)

X_bow_train <- as.matrix(dtm_train_bow_clean)
X_bow_test  <- as.matrix(dtm_test_bow)

# Explicit test-column alignment to match training vocabulary
missing_cols <- setdiff(colnames(X_bow_train), colnames(X_bow_test))

if (length(missing_cols) > 0) {
  zero_mat <- matrix(0, nrow = nrow(X_bow_test), ncol = length(missing_cols))
  colnames(zero_mat) <- missing_cols
  X_bow_test <- cbind(X_bow_test, zero_mat)
}

X_bow_test <- X_bow_test[, colnames(X_bow_train), drop = FALSE]


# =========================================================
# 10. BoW + MULTINOMIAL NAIVE BAYES
# =========================================================
# BoW features are raw counts -> multinomial distribution
# is the correct generative assumption for NB.

nb_bow_model <- multinomial_naive_bayes(
  x = X_bow_train,
  y = y_train,
  laplace = 1
)

nb_bow_pred <- predict(nb_bow_model, X_bow_test)
accuracy_bow_nb <- mean(nb_bow_pred == y_test)

bow_nb_result <- confusionMatrix(data = nb_bow_pred, reference = y_test)
metrics_bow_nb <- calc_macro_metrics(bow_nb_result)

bow_nb_precision <- metrics_bow_nb["precision"]
bow_nb_recall    <- metrics_bow_nb["recall"]
bow_nb_f1        <- metrics_bow_nb["f1"]

cat("\n=========================================\n")
cat(" 1. CONFUSION MATRIX: BoW + MULTINOMIAL NB\n")
cat("=========================================\n")
print(bow_nb_result)


# =========================================================
# 11. BoW + LOGISTIC REGRESSION
# =========================================================

set.seed(123)

logistic_bow_model <- cv.glmnet(
  X_bow_train,
  y_train,
  family = "multinomial"
)

bow_y_pred <- predict(
  logistic_bow_model,
  X_bow_test,
  type = "class",
  s = "lambda.min"
)

bow_y_pred <- factor(
  as.vector(bow_y_pred),
  levels = c("negative", "neutral", "positive")
)

bow_lr_result <- confusionMatrix(data = bow_y_pred, reference = y_test)
bow_accuracy_lr <- bow_lr_result$overall["Accuracy"]

metrics_bow_lr <- calc_macro_metrics(bow_lr_result)
bow_lr_precision <- metrics_bow_lr["precision"]
bow_lr_recall    <- metrics_bow_lr["recall"]
bow_lr_f1        <- metrics_bow_lr["f1"]

cat("\n=========================================\n")
cat(" 2. CONFUSION MATRIX: BoW + LOGISTIC REG\n")
cat("=========================================\n")
print(bow_lr_result)


# =========================================================
# 12. TF-IDF FEATURE EXTRACTION
# =========================================================
# FIX: use SUBLINEAR TF instead of L1-normalized TF.
# This is the canonical TF-IDF used in IR/NLP:
#   tf(t,d) = 1 + log(count(t,d))  for count > 0, else 0
#   idf(t)  = log2((N + 1) / (df(t) + 1)) + 1   (smoothed)
# idf is computed ONLY on training data (no leakage).

# Sublinear TF (element-wise; zeros stay zero)
tf_train <- X_bow_train
tf_train[tf_train > 0] <- 1 + log(tf_train[tf_train > 0])

tf_test <- X_bow_test
tf_test[tf_test > 0] <- 1 + log(tf_test[tf_test > 0])

# Smoothed, strictly-positive IDF from training set only
doc_freq     <- colSums(X_bow_train > 0)
n_train_docs <- nrow(X_bow_train)
idf_vector   <- log2((n_train_docs + 1) / (doc_freq + 1)) + 1

X_tfidf_train <- sweep(tf_train, 2, idf_vector, "*")
X_tfidf_test  <- sweep(tf_test[, names(idf_vector), drop = FALSE],
                       2, idf_vector, "*")


# =========================================================
# 13. TF-IDF + GAUSSIAN NAIVE BAYES
# =========================================================
# TF-IDF features are continuous -> Gaussian NB is the
# appropriate distributional assumption.
# (Note: many features are zero-variance within a class for
# sparse text; gaussian_naive_bayes handles this internally
# with a small variance floor, so results remain stable.)

nb_tfidf_model <- gaussian_naive_bayes(
  x = X_tfidf_train,
  y = y_train
)

nb_tfidf_pred <- predict(nb_tfidf_model, X_tfidf_test)
accuracy_tfidf_nb <- mean(nb_tfidf_pred == y_test)

tfidf_nb_result <- confusionMatrix(data = nb_tfidf_pred, reference = y_test)
metrics_tfidf_nb <- calc_macro_metrics(tfidf_nb_result)

tfidf_nb_precision <- metrics_tfidf_nb["precision"]
tfidf_nb_recall    <- metrics_tfidf_nb["recall"]
tfidf_nb_f1        <- metrics_tfidf_nb["f1"]

cat("\n=========================================\n")
cat(" 3. CONFUSION MATRIX: TF-IDF + GAUSSIAN NB\n")
cat("=========================================\n")
print(tfidf_nb_result)


# =========================================================
# 14. TF-IDF + LOGISTIC REGRESSION
# =========================================================

set.seed(123)

logistic_tfidf_model <- cv.glmnet(
  X_tfidf_train,
  y_train,
  family = "multinomial"
)

tfidf_y_pred <- predict(
  logistic_tfidf_model,
  X_tfidf_test,
  type = "class",
  s = "lambda.min"
)

tfidf_y_pred <- factor(
  as.vector(tfidf_y_pred),
  levels = c("negative", "neutral", "positive")
)

tfidf_lr_result <- confusionMatrix(data = tfidf_y_pred, reference = y_test)
accuracy_tfidf_lr <- tfidf_lr_result$overall["Accuracy"]

metrics_tfidf_lr <- calc_macro_metrics(tfidf_lr_result)
tfidf_lr_precision <- metrics_tfidf_lr["precision"]
tfidf_lr_recall    <- metrics_tfidf_lr["recall"]
tfidf_lr_f1        <- metrics_tfidf_lr["f1"]

cat("\n=========================================\n")
cat(" 4. CONFUSION MATRIX: TF-IDF + LOGISTIC REG\n")
cat("=========================================\n")
print(tfidf_lr_result)


# =========================================================
# 15. RESULTS SUMMARY TABLE
# =========================================================

comparison <- data.frame(
  Model_Type = c(
    "BoW + Multinomial NB",
    "BoW + Logistic Reg.",
    "TF-IDF + Gaussian NB",
    "TF-IDF + Logistic Reg."
  ),
  Accuracy  = c(as.numeric(accuracy_bow_nb),   as.numeric(bow_accuracy_lr),
                as.numeric(accuracy_tfidf_nb), as.numeric(accuracy_tfidf_lr)),
  Precision = c(bow_nb_precision, bow_lr_precision,
                tfidf_nb_precision, tfidf_lr_precision),
  Recall    = c(bow_nb_recall, bow_lr_recall,
                tfidf_nb_recall, tfidf_lr_recall),
  F1_Score  = c(bow_nb_f1, bow_lr_f1, tfidf_nb_f1, tfidf_lr_f1)
)

cat("\n=========================================\n")
cat("         MODEL COMPARISON SUMMARY\n")
cat("=========================================\n")

# Safer printing: round numeric cols only
comparison_print <- comparison
num_cols <- sapply(comparison_print, is.numeric)
comparison_print[num_cols] <- round(comparison_print[num_cols], 4)
print(comparison_print, row.names = FALSE)


# =========================================================
# 16. Majority-Class Baseline
# =========================================================

majority_class <- names(which.max(table(y_train)))
baseline_acc   <- mean(y_test == majority_class)

cat("\n=========================================\n")
cat("         BASELINE INFORMATION\n")
cat("=========================================\n")
cat("Majority class from training:", majority_class, "\n")
cat("Baseline test accuracy:", round(baseline_acc, 4), "\n")


# =========================================================
# 17. VISUALIZATIONS
# =========================================================

metrics_matrix <- t(as.matrix(comparison[, c("Accuracy", "Precision", "Recall", "F1_Score")]))
colnames(metrics_matrix) <- c(
  "BoW +\nMultinomial NB",
  "BoW +\nLogistic Reg.",
  "TF-IDF +\nGaussian NB",
  "TF-IDF +\nLogistic Reg."
)

bp <- barplot(
  metrics_matrix,
  beside = TRUE,
  col = c("#2B5C8F", "#E69F00", "#338A5E", "#B23B3B"),
  ylim = c(0, 1.1),
  main = "Model Performance Comparison",
  ylab = "Score",
  cex.names = 0.8
)

abline(h = baseline_acc, col = "gray40", lty = 2, lwd = 2)

legend(
  "bottom",
  legend = c("Majority-class baseline", "Accuracy", "Precision", "Recall", "F1-Score"),
  col = c("gray40", "#2B5C8F", "#E69F00", "#338A5E", "#B23B3B"),
  lty = c(2, NA, NA, NA, NA),
  lwd = c(2, NA, NA, NA, NA),
  pch = c(NA, 15, 15, 15, 15),
  pt.cex = 1.5,
  bty = "n",
  horiz = TRUE,
  xpd = TRUE,
  inset = c(0, -0.25)
)


# --- Line Graph ---
plot(
  comparison$Accuracy,
  type = "o",
  col = "blue",
  pch = 16,
  ylim = c(0, 1),
  xaxt = "n",
  xlab = "Model Approach",
  ylab = "Metric Score",
  main = "Line Graph: Performance Comparison"
)

axis(
  1,
  at = 1:4,
  labels = c("BoW+MNB", "BoW+LR", "TFIDF+GNB", "TFIDF+LR"),
  cex.axis = 0.8
)

lines(comparison$F1_Score, type = "o", col = "red", pch = 17)

legend(
  "bottomright",
  legend = c("Accuracy", "F1 Score"),
  col = c("blue", "red"),
  pch = c(16, 17),
  lty = 1
)


# =========================================================
# 18. RADAR CHART - ALL FOUR MODELS
# =========================================================

radar_data <- as.data.frame(
  rbind(
    rep(1, 4),
    rep(0, 4),
    comparison[, c("Accuracy", "Precision", "Recall", "F1_Score")]
  )
)

colnames(radar_data) <- c("Accuracy", "Precision", "Recall", "F1 Score")
rownames(radar_data) <- c("Max", "Min", comparison$Model_Type)

radar_colors <- c("#2B5C8F", "#E69F00", "#338A5E", "#B23B3B")

radarchart(
  radar_data,
  axistype = 1,
  pcol = radar_colors,
  plwd = 2,
  plty = 1,
  cglcol = "grey",
  cglty = 1,
  axislcol = "grey",
  caxislabels = seq(0, 1, 0.25),
  cglwd = 0.8,
  title = "Radar Chart: All Models Compared"
)

legend(
  "topright",
  legend = comparison$Model_Type,
  col = radar_colors,
  lty = 1,
  lwd = 2,
  cex = 0.6,
  bty = "n"
)

par(mfrow = c(1, 1))


# =========================================================
# 19. FINAL OUTPUT
# =========================================================

cat("\n=========================================\n")
cat("              FINAL RESULTS\n")
cat("=========================================\n")
print(comparison_print, row.names = FALSE)

cat("\nAnalysis completed successfully.\n")

