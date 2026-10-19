# =============================================================================
# Spam detection: comparing supervised learning methods on the Spambase data
# -----------------------------------------------------------------------------
# Data   : UCI Spambase (4,601 e-mails, 57 word/character-frequency predictors),
#          shipped as `spam` in the kernlab package.
# Goal   : predict spam vs non-spam; compare 10 classifiers on a held-out test
#          set, tuning every hyper-parameter by 5-fold cross-validation on the
#          training set only.
# Run    : Rscript spam_classification.R   (writes results/ and figures/)
# =============================================================================

pkgs <- c("kernlab", "class", "MASS", "e1071", "glmnet", "rpart",
          "randomForest", "gbm")
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")
invisible(lapply(pkgs, library, character.only = TRUE))

dir.create("results", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

# ---- Data and train/test split ------------------------------------------------
data(spam, package = "kernlab")
set.seed(1234)
n        <- nrow(spam)
train_id <- sample(n, size = round(2 / 3 * n))
train    <- spam[train_id, ]
test     <- spam[-train_id, ]

x.tr <- as.matrix(train[, -58]); y.tr <- train$type
x.te <- as.matrix(test[, -58]);  y.te <- test$type

# Standardise with TRAINING statistics only (no test-set leakage)
mu.tr    <- colMeans(x.tr)
sd.tr    <- apply(x.tr, 2, sd)
x.tr.std <- scale(x.tr, mu.tr, sd.tr)
x.te.std <- scale(x.te, mu.tr, sd.tr)

# Same 5 folds reused by every hand-written CV loop
folds <- sample(rep(1:5, length.out = nrow(x.tr)))

err <- function(yhat, y) mean(as.character(yhat) != as.character(y))
results <- data.frame(method = character(), tuning = character(),
                      cv_error = numeric(), test_error = numeric())
add <- function(method, tuning, cv, test) {
  results[nrow(results) + 1, ] <<- list(method, tuning, round(cv, 4), round(test, 4))
  cat(sprintf("%-26s %-22s CV %.4f  test %.4f\n", method, tuning, cv, test))
}

# ---- 1. Logistic regression -----------------------------------------------------
cv.logit <- sapply(1:5, function(i) {
  fit <- suppressWarnings(glm(type ~ ., data = train[folds != i, ], family = binomial))
  p   <- predict(fit, train[folds == i, ], type = "response")
  err(ifelse(p > 0.5, "spam", "nonspam"), y.tr[folds == i])
})
logit <- suppressWarnings(glm(type ~ ., data = train, family = binomial))
p.te  <- predict(logit, test, type = "response")
add("Logistic regression", "-", mean(cv.logit),
    err(ifelse(p.te > 0.5, "spam", "nonspam"), y.te))

# ---- 2. LASSO logistic regression ----------------------------------------------
cv.lasso <- cv.glmnet(x.tr.std, y.tr, family = "binomial", alpha = 1,
                      type.measure = "class", foldid = folds)
yhat     <- predict(cv.lasso, x.te.std, s = "lambda.1se", type = "class")
n.nonzero <- sum(coef(cv.lasso, s = "lambda.1se")[-1] != 0)
add("LASSO logistic", sprintf("lambda.1se, %d vars", n.nonzero),
    cv.lasso$cvm[cv.lasso$lambda == cv.lasso$lambda.1se], err(yhat, y.te))

png("figures/lasso_cv.png", 900, 600, res = 120)
plot(cv.lasso); title("LASSO: 5-fold CV misclassification error", line = 2.5)
invisible(dev.off())

# ---- 3. Linear discriminant analysis -------------------------------------------
cv.lda <- sapply(1:5, function(i) {
  fit <- lda(type ~ ., data = train[folds != i, ])
  err(predict(fit, train[folds == i, ])$class, y.tr[folds == i])
})
lda.fit <- lda(type ~ ., data = train)
add("LDA", "-", mean(cv.lda), err(predict(lda.fit, test)$class, y.te))

# ---- 4. Naive Bayes (Gaussian) --------------------------------------------------
cv.nb <- sapply(1:5, function(i) {
  fit <- naiveBayes(x.tr.std[folds != i, ], y.tr[folds != i])
  err(predict(fit, x.tr.std[folds == i, ]), y.tr[folds == i])
})
nb <- naiveBayes(x.tr.std, y.tr)
add("Naive Bayes", "Gaussian", mean(cv.nb), err(predict(nb, x.te.std), y.te))

# ---- 5. K-nearest neighbours (k chosen by CV) ----------------------------------
k.grid <- c(1, 3, 5, 7, 9, 15, 21)
cv.knn <- sapply(k.grid, function(k) mean(sapply(1:5, function(i)
  err(knn(x.tr.std[folds != i, ], x.tr.std[folds == i, ], y.tr[folds != i], k = k),
      y.tr[folds == i]))))
best.k <- k.grid[which.min(cv.knn)]
set.seed(1234)
add("KNN", paste0("k = ", best.k), min(cv.knn),
    err(knn(x.tr.std, x.te.std, y.tr, k = best.k), y.te))

png("figures/knn_cv.png", 900, 600, res = 120)
plot(k.grid, cv.knn, type = "b", pch = 19, xlab = "k", ylab = "5-fold CV error",
     main = "KNN: choosing k by cross-validation")
abline(v = best.k, lty = 2, col = "grey40")
invisible(dev.off())

# ---- 6. Classification tree (pruned by CV) -------------------------------------
big.tree <- rpart(type ~ ., data = train, method = "class",
                  control = rpart.control(cp = 0, xval = 5))
cp.tab   <- big.tree$cptable
# one-standard-error rule
thr      <- min(cp.tab[, "xerror"]) + cp.tab[which.min(cp.tab[, "xerror"]), "xstd"]
best.cp  <- cp.tab[which(cp.tab[, "xerror"] <= thr)[1], "CP"]
pruned   <- prune(big.tree, cp = best.cp)
tree.cv  <- cp.tab[cp.tab[, "CP"] == best.cp, "xerror"] * mean(y.tr == "spam")
add("Classification tree", sprintf("%d leaves (1-SE)", sum(pruned$frame$var == "<leaf>")),
    tree.cv, err(predict(pruned, test, type = "class"), y.te))

png("figures/pruned_tree.png", 1100, 750, res = 120)
plot(pruned, uniform = TRUE, margin = 0.05); text(pruned, cex = 0.6, use.n = FALSE)
title("Pruned classification tree")
invisible(dev.off())

# ---- 7. Bagging and 8. Random forest -------------------------------------------
set.seed(1234)
bag <- randomForest(type ~ ., data = train, mtry = 57, ntree = 500, importance = TRUE)
add("Bagging", "mtry = 57, 500 trees", tail(bag$err.rate[, "OOB"], 1),
    err(predict(bag, test), y.te))

set.seed(1234)
rf <- randomForest(type ~ ., data = train, ntree = 500, importance = TRUE)
add("Random forest", sprintf("mtry = %d, 500 trees", rf$mtry),
    tail(rf$err.rate[, "OOB"], 1), err(predict(rf, test), y.te))

png("figures/rf_importance.png", 1000, 700, res = 120)
varImpPlot(rf, n.var = 15, main = "Random forest: top 15 predictors")
invisible(dev.off())

# ---- 9. Gradient boosting (number of trees chosen by CV) -----------------------
train.b <- transform(train, type = as.integer(type == "spam"))
set.seed(1234)
boost <- gbm(type ~ ., data = train.b, distribution = "bernoulli", n.trees = 3000,
             interaction.depth = 4, shrinkage = 0.05, cv.folds = 5, verbose = FALSE)
best.trees <- gbm.perf(boost, method = "cv", plot.it = FALSE)
p.cv       <- 1 / (1 + exp(-boost$cv.fitted))
p.te       <- predict(boost, test, n.trees = best.trees, type = "response")
add("Boosting (gbm)", sprintf("%d trees, depth 4", best.trees),
    err(ifelse(p.cv > 0.5, 1, 0), train.b$type),
    err(ifelse(p.te > 0.5, "spam", "nonspam"), y.te))

# ---- 10. Support vector machine (radial kernel, tuned by CV) -------------------
set.seed(1234)
svm.tune <- tune(svm, train.x = x.tr.std, train.y = y.tr, kernel = "radial",
                 ranges = list(cost = c(1, 10, 100, 1000),
                               gamma = c(1e-4, 5e-4, 1e-3, 5e-3, 0.01)),
                 tunecontrol = tune.control(cross = 5))
best.svm <- svm.tune$best.model
add("SVM (radial)", sprintf("C = %g, gamma = %g", svm.tune$best.parameters$cost,
                            svm.tune$best.parameters$gamma),
    svm.tune$best.performance, err(predict(best.svm, x.te.std), y.te))

# ---- Summary --------------------------------------------------------------------
results <- results[order(results$test_error), ]
write.csv(results, "results/model_comparison.csv", row.names = FALSE)
print(results, row.names = FALSE)

png("figures/model_comparison.png", 1000, 650, res = 120)
par(mar = c(4, 11, 3, 1))
barplot(rev(results$test_error), names.arg = rev(results$method), horiz = TRUE,
        las = 1, xlab = "Test misclassification error",
        main = "Spambase: test error by method", col = "steelblue")
invisible(dev.off())
