context("posterior weight engine")
library(logitr)

panel_data_weights <- subset(yogurt, id %in% 1:8 & obsID <= 150)

panel_model_weights <- logitr(
  data = panel_data_weights,
  outcome = "choice",
  obsID = "obsID",
  pars = c("price", "feat", "brand"),
  randPars = c(price = "n"),
  panelID = "id",
  numDraws = 20
)

holdout_weights <- data.frame(
  id = c(1, 1, 1, 1, 2, 2, 2, 2),
  obsID = c(1001, 1001, 1001, 1001, 1002, 1002, 1002, 1002),
  price = c(0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4),
  feat = c(0, 1, 0, 1, 0, 1, 0, 1),
  brand = rep(c("dannon", "hiland", "weight", "yoplait"), 2),
  stringsAsFactors = FALSE
)

test_that("posterior weight engine returns normalized weights aligned with fitted draws", {
  engine <- makePosteriorWeightEngine(panel_model_weights)

  expect_equal(engine$panelIDs, panel_model_weights$data$panelIDValues)
  expect_identical(rownames(engine$weights), as.character(engine$panelIDs))
  expect_equal(
    dim(engine$weights),
    c(length(engine$panelIDs), panel_model_weights$n$draws)
  )
  expect_equal(
    as.numeric(rowSums(engine$weights)),
    rep(1, length(engine$panelIDs)),
    tolerance = 1e-8
  )
})

test_that("posterior weight engine returns finite non-negative weights", {
  engine <- makePosteriorWeightEngine(panel_model_weights)

  expect_true(all(is.finite(engine$weights)))
  expect_true(all(engine$weights >= 0))
})

test_that("posterior weight engine matches the Revelt-Train draw-weight formula", {
  predict_funcs <- getPredictVFunctions(panel_model_weights$modelSpace)
  data_diff <- makeDiffData(
    panel_model_weights$data, panel_model_weights$modelType
  )
  beta_draws <- makeBetaDraws(
    stats::coef(panel_model_weights),
    panel_model_weights$parIDs,
    panel_model_weights$n,
    panel_model_weights$standardDraws,
    panel_model_weights$inputs$correlation
  )
  colnames(beta_draws) <- names(panel_model_weights$parSetup)
  beta_draws <- selectDraws(
    beta_draws, panel_model_weights$modelSpace, data_diff$X
  )
  v_draws <- predict_funcs$getVDraws(
    beta_draws, data_diff$X, data_diff$scalePar, panel_model_weights$n
  )
  p_y_given_beta <- getLogit(exp(v_draws), data_diff$obsID)
  manual_log_weights <- rowsum(
    log(p_y_given_beta),
    group = data_diff$panelID,
    reorder = FALSE
  )
  manual_log_weights <- sweep(
    manual_log_weights, 1, apply(manual_log_weights, 1, max), "-"
  )
  manual_weights <- exp(manual_log_weights)
  manual_weights <- manual_weights / rowSums(manual_weights)

  engine <- makePosteriorWeightEngine(panel_model_weights)
  expect_equal(engine$weights, manual_weights, tolerance = 1e-10)
})

test_that("conditional holdout probabilities match the weighted draw formula", {
  engine <- makePosteriorWeightEngine(panel_model_weights)
  holdout_data <- formatNewData(
    panel_model_weights, holdout_weights, "obsID", "id"
  )
  panel_index <- matchPosteriorWeightPanelIDs(
    engine, holdout_data$panelID, "id"
  )
  holdout_draws <- getPredictionDraws(panel_model_weights, holdout_data)
  manual_probs <- rowSums(
    holdout_draws * engine$weights[panel_index, , drop = FALSE]
  )

  conditional_probs <- predict(
    panel_model_weights,
    newdata = holdout_weights,
    obsID = "obsID",
    panelID = "id",
    conditional = TRUE
  )

  expect_equal(
    unname(conditional_probs$predicted_prob),
    unname(manual_probs),
    tolerance = 1e-10
  )
})
