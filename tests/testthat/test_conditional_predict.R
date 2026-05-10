context("conditional_predict")
library(logitr)

model <- readRDS(system.file("extdata", "mxl_pref.Rds", package = "logitr"))
mxl_wtp_model <- readRDS(system.file("extdata", "mxl_wtp.Rds", package = "logitr"))

conditioning_data <- subset(yogurt, id %in% c(1, 2))

task_template <- subset(
  yogurt,
  obsID == 1,
  select = c("obsID", "id", "alt", "price", "feat", "brand")
)

newdata <- rbind(
  transform(task_template, obsID = 9001, id = 1),
  transform(task_template, obsID = 9002, id = 2)
)

test_that("predict() remains unconditional by default for identical new tasks", {
  p <- predict(
    model,
    newdata = newdata,
    obsID = "obsID"
  )

  probs_1 <- subset(p, obsID == 9001)$predicted_prob
  probs_2 <- subset(p, obsID == 9002)$predicted_prob
  expect_equal(probs_1, probs_2)
})

test_that("conditional_predict() conditions predictions on known individuals", {
  p <- conditional_predict(
    model,
    conditioning_data = conditioning_data,
    newdata = newdata,
    returnData = TRUE
  )

  expect_equal(nrow(p), nrow(newdata))
  expect_true(all(c("obsID", "predicted_prob", "id") %in% names(p)))

  probs_1 <- subset(p, id == 1)$predicted_prob
  probs_2 <- subset(p, id == 2)$predicted_prob
  expect_false(isTRUE(all.equal(probs_1, probs_2)))
})

test_that("conditional behavior only activates through conditional_predict()", {
  unconditional <- predict(
    model,
    newdata = newdata,
    obsID = "obsID"
  )
  conditional <- conditional_predict(
    model,
    conditioning_data = conditioning_data,
    newdata = newdata
  )

  unconditional_1 <- subset(unconditional, obsID == 9001)$predicted_prob
  unconditional_2 <- subset(unconditional, obsID == 9002)$predicted_prob
  conditional_1 <- conditional$predicted_prob[newdata$id == 1]
  conditional_2 <- conditional$predicted_prob[newdata$id == 2]

  expect_equal(unconditional_1, unconditional_2)
  expect_false(isTRUE(all.equal(conditional_1, conditional_2)))
})

test_that("conditional_predict() errors for unknown individuals in newdata", {
  unknown_newdata <- transform(task_template, obsID = 9003, id = 3)

  expect_error(
    conditional_predict(
      model,
      conditioning_data = conditioning_data,
      newdata = unknown_newdata
    ),
    "panelID"
  )
})

test_that("conditional_predict() accepts short histories and ignores extra conditioning individuals", {
  short_conditioning_data <- subset(
    yogurt,
    (id == 1 & obsID == 1) | id == 2
  )
  one_id_newdata <- transform(task_template, obsID = 9004, id = 1)

  p <- conditional_predict(
    model,
    conditioning_data = short_conditioning_data,
    newdata = one_id_newdata,
    returnData = TRUE
  )

  expect_equal(nrow(p), nrow(one_id_newdata))
  expect_true(all(p$id == 1))
  expect_true(all(is.finite(p$predicted_prob)))
})

test_that("conditional_predict() is reproducible and finite for longer panel histories", {
  long_conditioning_data <- subset(yogurt, id == 1)
  long_newdata <- transform(task_template, obsID = 9005, id = 1)

  p1 <- conditional_predict(
    model,
    conditioning_data = long_conditioning_data,
    newdata = long_newdata
  )
  p2 <- conditional_predict(
    model,
    conditioning_data = long_conditioning_data,
    newdata = long_newdata
  )

  expect_equal(p1, p2)
  expect_true(all(is.finite(p1$predicted_prob)))
  expect_equal(sum(p1$predicted_prob), 1, tolerance = 1e-8)
})

test_that("conditional_predict() is reproducible across seed changes", {
  set.seed(1)
  p1 <- conditional_predict(
    model,
    conditioning_data = conditioning_data,
    newdata = newdata
  )
  set.seed(999)
  p2 <- conditional_predict(
    model,
    conditioning_data = conditioning_data,
    newdata = newdata
  )

  expect_equal(p1, p2)
})

test_that("conditional_means() returns one row per individual and random-parameter columns", {
  means <- conditional_means(
    model,
    conditioning_data = conditioning_data
  )

  expect_equal(nrow(means), 2)
  expect_true(all(c("id", "feat", "brandhiland", "brandweight",
                    "brandyoplait") %in% names(means)))
  expect_false("price" %in% names(means))
})

test_that("conditional_means() agrees with direct posterior-weighted averages", {
  one_id_conditioning <- subset(yogurt, id == 1)
  means <- conditional_means(
    model,
    conditioning_data = one_id_conditioning
  )

  recoded <- recodeData(one_id_conditioning, model$inputs$pars, model$inputs$randPars)
  X <- recoded$X
  beta_draws <- get("makeBetaDraws", envir = asNamespace("logitr"))(
    coef(model), model$parIDs, model$n, model$standardDraws, model$inputs$correlation
  )
  colnames(beta_draws) <- names(model$parSetup)
  V <- X %*% t(beta_draws)
  expV <- exp(V)
  sum_expV <- rowsum(expV, group = one_id_conditioning$obsID, reorder = FALSE)
  reps <- table(one_id_conditioning$obsID)
  logit_draws <- expV / sum_expV[rep(seq_along(reps), reps), ]
  chosen_draws <- logit_draws[one_id_conditioning$choice == 1, , drop = FALSE]
  log_weights <- colSums(log(chosen_draws))
  log_weights <- log_weights - max(log_weights)
  weights <- exp(log_weights)
  weights <- weights / sum(weights)
  expected <- colSums(beta_draws[, model$parIDs$r, drop = FALSE] * weights)

  expect_equal(
    unname(as.numeric(means[1, c("feat", "brandhiland", "brandweight", "brandyoplait")])),
    unname(as.numeric(expected)),
    tolerance = 1e-8
  )
})

test_that("conditional_predict() works for WTP-space mixed logit models", {
  p <- conditional_predict(
    mxl_wtp_model,
    conditioning_data = conditioning_data,
    newdata = newdata,
    returnData = TRUE
  )

  expect_equal(nrow(p), nrow(newdata))
  expect_true(all(c("obsID", "predicted_prob", "id", "price") %in% names(p)))
  expect_true(all(is.finite(p$predicted_prob)))
})

test_that("conditional_means() works for WTP-space mixed logit models", {
  means <- conditional_means(
    mxl_wtp_model,
    conditioning_data = conditioning_data
  )

  expect_equal(nrow(means), 2)
  expect_true(all(c("id", "feat", "brandhiland", "brandweight",
                    "brandyoplait") %in% names(means)))
  expect_false("scalePar" %in% names(means))
})
