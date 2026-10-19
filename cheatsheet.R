# =============================================================================
# Supervised statistical learning in R: a practical cheat sheet
# -----------------------------------------------------------------------------
# Notation: x  = n x p predictor matrix / data frame
#           y  = response (factor for classification, numeric for regression)
#           dtf = one data frame holding y and the predictors
# =============================================================================

# ---- Data format quick reference ---------------------------------------------
# Formula interface (ONE combined data frame):
#   glm(), lda(), rpart()/tree(), randomForest(), gbm(), svm(), pcr()
#   dtf <- data.frame(y = as.factor(y), x)   # classification
#   dtf <- data.frame(y = y, x)              # regression
# Matrix interface (x and y SEPARATE):
#   knn(), naiveBayes(), glmnet(), regsubsets() (also has a formula method)

# ---- When to standardise --------------------------------------------------------
# YES : KNN, SVM, ridge, lasso, PCR, PLS (distance-, kernel- or penalty-based)
# NO  : trees, bagging, random forests, boosting, LDA, naive Bayes
# Always standardise with TRAINING statistics to avoid test-set leakage:
#   mu.tr    <- apply(x.tr, 2, mean)
#   sd.tr    <- apply(x.tr, 2, sd)
#   x.tr.std <- scale(x.tr, mu.tr, sd.tr)
#   x.te.std <- scale(x.te, mu.tr, sd.tr)   # NOT scale(x.te, TRUE, TRUE)

# ---- Helpers --------------------------------------------------------------------
misc <- function(yhat, y) {                 # misclassification rate
  if (length(table(yhat)) != length(table(y)))
    stop("The levels of the two vectors do not match")
  1 - sum(diag(table(yhat, y))) / length(y)
}

predict.regsubsets <- function(object, newdata, id, ...) {
  form  <- as.formula(object$call[[2]])
  mat   <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  mat[, names(coefi)] %*% coefi
}

# K-fold assignment, reused by every CV loop so methods are comparable
set.seed(1234)
folds <- sample(rep(1:5, length.out = nrow(x.tr)))

# ---- Logistic regression ------------------------------------------------------
glm(y ~ ., data = dtf, family = "binomial")

# ---- Naive Bayes ----------------------------------------------------------------
# Assumes predictors are conditionally independent given the class, so the joint
# likelihood factorises into univariate densities: tractable in high dimension.
# No hyper-parameter to tune: CV error is only an estimate, not a selection tool.
library(e1071)                              # or klaR::NaiveBayes()
nb   <- naiveBayes(x.tr, y.tr)
yhat <- predict(nb, x.te)

# ---- K-nearest neighbours -----------------------------------------------------
# Distance-based: standardise. Use CV to choose k, then refit on all training data.
library(class)
k      <- c(1, 3, 5, 7, 9)
err.cv <- matrix(NA, 5, length(k), dimnames = list(NULL, paste0("K=", k)))
for (i in 1:5) {
  for (j in seq_along(k)) {
    yhat         <- knn(x.tr.std[folds != i, ], x.tr.std[folds == i, ],
                        y.tr[folds != i], k = k[j])
    err.cv[i, j] <- mean(yhat != y.tr[folds == i])
  }
}
best.k <- k[which.min(colMeans(err.cv))]
yhat   <- knn(x.tr.std, x.te.std, y.tr, k = best.k)
mean(yhat != y.te)                          # test error

# ---- Linear discriminant analysis ----------------------------------------------
library(MASS)
lda(y ~ ., data = dtf)

# ---- Subset selection ------------------------------------------------------------
library(leaps)
regsubsets(y ~ ., data = dtf)                       # best subset
regsubsets(y ~ ., data = dtf, method = "forward")   # forward stepwise
regsubsets(y ~ ., data = dtf, method = "backward")  # backward stepwise
regsubsets(y ~ ., data = dtf, method = "seqrep")    # hybrid

# ---- Ridge and lasso ------------------------------------------------------------
# x must be a matrix; the penalty treats all coefficients equally, so standardise.
library(glmnet)
cv.glmnet(x, y, alpha = 0)                  # ridge, lambda by CV
cv.glmnet(x, y, alpha = 1)                  # lasso, lambda by CV

# ---- Principal component regression -------------------------------------------
library(pls)
pcr.fit <- pcr(y ~ ., data = dtf, scale = TRUE, validation = "CV")
validationplot(pcr.fit)
selectNcomp(pcr.fit, method = "onesigma", plot = TRUE)

# ---- Trees ------------------------------------------------------------------------
# Scale-invariant splits: no standardisation. y must be a factor to classify.
# The root split is the single most important variable.
library(tree)
clas.tree <- tree(y ~ ., dtf)
cv.out    <- cv.tree(clas.tree, FUN = prune.misclass)
pruned    <- prune.misclass(clas.tree, best = cv.out$size[which.min(cv.out$dev)])
# Regression trees: tree() + cv.tree(FUN = prune.tree) + prune.tree()

# ---- Bagging, random forests, boosting ----------------------------------------
# Bagging = random forest with mtry = p: averages many bootstrap trees to cut the
# variance of a single deep tree. Random forest (mtry = sqrt(p) for
# classification) also decorrelates the trees.
library(randomForest)
bag <- randomForest(y ~ ., data = dtf, mtry = p, importance = TRUE)
rf  <- randomForest(y ~ ., data = dtf, importance = TRUE)
varImpPlot(rf)

library(gbm)                                # y coded 0/1 for "bernoulli"
boost <- gbm(y ~ ., data = dtf, distribution = "bernoulli", n.trees = 1000,
             interaction.depth = 4, shrinkage = 0.05, cv.folds = 5)
gbm.perf(boost, method = "cv")              # number of trees chosen by CV

# ---- Support vector machines ----------------------------------------------------
library(e1071)
svm(y ~ ., data = dtf, kernel = "linear", cost = 10)
svm(y ~ ., data = dtf, kernel = "radial", gamma = 1, cost = 1)
svm(y ~ ., data = dtf, kernel = "polynomial", degree = 3, cost = 1)
tune(svm, y ~ ., data = dtf,
     ranges = list(cost = c(0.1, 1, 10, 100, 1000),
                   kernel = c("linear", "polynomial", "radial")))

# ---- Comparing models -------------------------------------------------------------
# CV error  -> choose hyper-parameters within a method (k, lambda, tree size ...)
# Test error -> compare different methods, once, at the end.
# Trade-off: a single tree is easy to interpret but less accurate; bagging,
# random forests and boosting are more accurate but harder to interpret.
