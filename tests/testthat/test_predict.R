context("predict method")
library(logitr)

model <- logitr(
    data    = yogurt,
    outcome = 'choice',
    obsID   = 'obsID',
    pars    = c('price', 'feat', 'brand')
)

data <- subset(
  yogurt, obsID %in% c(42, 13),
  select = c('obsID', 'alt', 'price', 'feat', 'brand'))

panel_data <- subset(yogurt, id %in% 1:8 & obsID <= 150)

panel_model <- logitr(
  data    = panel_data,
  outcome = "choice",
  obsID   = "obsID",
  pars    = c("price", "feat", "brand"),
  randPars = c(price = "n"),
  panelID = "id",
  numDraws = 20
)

mixed_model_no_panel <- logitr(
  data    = panel_data,
  outcome = "choice",
  obsID   = "obsID",
  pars    = c("price", "feat", "brand"),
  randPars = c(price = "n"),
  numDraws = 20
)

wtp_panel_model <- logitr(
  data = panel_data,
  outcome = "choice",
  obsID = "obsID",
  pars = c("feat", "brand"),
  scalePar = "price",
  randPars = c(feat = "n", brand = "n"),
  panelID = "id",
  numDraws = 20
)

conditional_holdout <- data.frame(
  id = c(1, 1, 1, 1, 2, 2, 2, 2),
  obsID = c(1001, 1001, 1001, 1001, 1002, 1002, 1002, 1002),
  alt = rep(c(1, 2, 3, 4), 2),
  price = c(0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4),
  feat = c(0, 1, 0, 1, 0, 1, 0, 1),
  brand = rep(c("dannon", "hiland", "weight", "yoplait"), 2),
  stringsAsFactors = FALSE
)

encoded_conditional_holdout <- data.frame(
  id = conditional_holdout$id,
  obsID = conditional_holdout$obsID,
  recodeData(
    conditional_holdout,
    c("price", "feat", "brand"),
    c(price = "n")
  )$X,
  check.names = FALSE
)

test_that("predict() uses model data if newdata == NULL", {
  p <- predict(model)
  expect_equal(nrow(p), nrow(yogurt))
})

test_that("predict() uses newdata if provided", {
  p <- predict(model, newdata = data, obsID = "obsID")
  expect_equal(nrow(p), nrow(data))
})

test_that("predict() returns the correct user-specified prediction types", {
  x <- predict(model, newdata = data, obsID = "obsID")
  expect_true(
    (! "predicted_outcome" %in% names(x)) &
    ("predicted_prob" %in% names(x))
  )
  x <- predict(model, newdata = data, obsID = "obsID", type = "prob")
  expect_true(
    (! "predicted_outcome" %in% names(x)) &
    ("predicted_prob" %in% names(x))
  )
  x <- predict(model, newdata = data, obsID = "obsID", type = "outcome")
  expect_true(
    ("predicted_outcome" %in% names(x)) &
    (! "predicted_prob" %in% names(x))
  )
  x <- predict(
    model, newdata = data, obsID = "obsID",
    type = c("outcome", "prob")
  )
  expect_true(
    ("predicted_outcome" %in% names(x)) &
    ("predicted_prob" %in% names(x))
  )
  x <- predict(
    model, newdata = data, obsID = "obsID",
    type = c("prob", "outcome")
  )
  expect_true(
    ("predicted_outcome" %in% names(x)) &
    ("predicted_prob" %in% names(x))
  )
})

test_that("predict() supports conditional holdout probabilities for known panel individuals", {
  population <- predict(
    panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    returnData = TRUE
  )
  conditional <- predict(
    panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    panelID = "id",
    conditional = TRUE,
    returnData = TRUE
  )

  expect_equal(nrow(conditional), nrow(conditional_holdout))
  expect_identical(names(conditional), names(population))
  expect_equal(
    population$predicted_prob[population$obsID == 1001],
    population$predicted_prob[population$obsID == 1002]
  )
  expect_false(isTRUE(all.equal(
    conditional$predicted_prob[conditional$obsID == 1001],
    conditional$predicted_prob[conditional$obsID == 1002]
  )))
  expect_equal(
    as.numeric(tapply(conditional$predicted_prob, conditional$obsID, sum)),
    c(1, 1),
    tolerance = 1e-8
  )
})

test_that("predict() accepts already-encoded holdout inputs in population and conditional mode", {
  raw_population <- predict(
    panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    returnData = TRUE
  )
  encoded_population <- predict(
    panel_model,
    newdata = encoded_conditional_holdout,
    obsID = "obsID",
    returnData = TRUE
  )
  expect_equal(encoded_population$predicted_prob, raw_population$predicted_prob)
  expect_true("brandhiland" %in% names(encoded_population))

  raw_conditional <- predict(
    panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    panelID = "id",
    conditional = TRUE,
    returnData = TRUE
  )
  encoded_conditional <- predict(
    panel_model,
    newdata = encoded_conditional_holdout,
    obsID = "obsID",
    panelID = "id",
    conditional = TRUE,
    returnData = TRUE
  )
  expect_equal(encoded_conditional$predicted_prob, raw_conditional$predicted_prob)
  expect_true("brandhiland" %in% names(encoded_conditional))
})

test_that("predict() supports conditional holdout probabilities for WTP-space mixed logit models", {
  population <- predict(
    wtp_panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    returnData = TRUE
  )
  conditional <- predict(
    wtp_panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    panelID = "id",
    conditional = TRUE,
    returnData = TRUE
  )

  expect_identical(names(conditional), names(population))
  expect_equal(
    population$predicted_prob[population$obsID == 1001],
    population$predicted_prob[population$obsID == 1002]
  )
  expect_false(isTRUE(all.equal(
    conditional$predicted_prob[conditional$obsID == 1001],
    conditional$predicted_prob[conditional$obsID == 1002]
  )))
  expect_equal(
    as.numeric(tapply(conditional$predicted_prob, conditional$obsID, sum)),
    c(1, 1),
    tolerance = 1e-8
  )
})

test_that("predict() preserves output-type handling and data passthrough in conditional mode", {
  population_prob <- predict(
    panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    type = "prob",
    returnData = TRUE
  )
  conditional_prob <- predict(
    panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    panelID = "id",
    conditional = TRUE,
    type = "prob",
    returnData = TRUE
  )
  expect_identical(names(conditional_prob), names(population_prob))
  expect_equal(conditional_prob$id, conditional_holdout$id)
  expect_equal(conditional_prob$brand, conditional_holdout$brand)

  population_outcome <- predict(
    panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    type = "outcome",
    returnData = TRUE
  )
  conditional_outcome <- predict(
    panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    panelID = "id",
    conditional = TRUE,
    type = "outcome",
    returnData = TRUE
  )
  expect_identical(names(conditional_outcome), names(population_outcome))
  expect_true("predicted_outcome" %in% names(conditional_outcome))
  expect_false("predicted_prob" %in% names(conditional_outcome))

  population_both <- predict(
    panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    type = c("prob", "outcome"),
    returnData = TRUE
  )
  conditional_both <- predict(
    panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    panelID = "id",
    conditional = TRUE,
    type = c("prob", "outcome"),
    returnData = TRUE
  )
  expect_identical(names(conditional_both), names(population_both))
})

test_that("predict() keeps default population predictions unchanged when conditional is not used", {
  baseline <- predict(
    panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    type = c("prob", "outcome"),
    returnData = TRUE
  )
  explicit_default <- predict(
    panel_model,
    newdata = conditional_holdout,
    obsID = "obsID",
    conditional = FALSE,
    type = c("prob", "outcome"),
    returnData = TRUE
  )

  expect_identical(names(explicit_default), names(baseline))
  expect_equal(explicit_default$predicted_prob, baseline$predicted_prob)
})

test_that("predict() rejects unseen panel individuals in conditional mode", {
  unseen_holdout <- conditional_holdout
  unseen_holdout$id[unseen_holdout$obsID == 1002] <- 999

  expect_error(
    predict(
      panel_model,
      newdata = unseen_holdout,
      obsID = "obsID",
      panelID = "id",
      conditional = TRUE
    ),
    'not seen during model estimation'
  )
})

test_that("predict() rejects confidence and prediction intervals in conditional mode", {
  expect_error(
    predict(
      panel_model,
      newdata = conditional_holdout,
      obsID = "obsID",
      panelID = "id",
      conditional = TRUE,
      interval = "confidence"
    ),
    'supports only interval = "none"'
  )

  expect_error(
    predict(
      panel_model,
      newdata = conditional_holdout,
      obsID = "obsID",
      panelID = "id",
      conditional = TRUE,
      interval = "prediction"
    ),
    'supports only interval = "none"'
  )
})

test_that("predict() requires a valid panelID argument in conditional mode", {
  expect_error(
    predict(
      panel_model,
      newdata = conditional_holdout,
      obsID = "obsID",
      conditional = TRUE
    ),
    '"panelID" must be specified when conditional = TRUE'
  )

  expect_error(
    predict(
      panel_model,
      newdata = conditional_holdout,
      obsID = "obsID",
      panelID = "missing_id",
      conditional = TRUE
    ),
    'refers to a column that does not exist'
  )
})

test_that("predict() rejects unsupported models in conditional mode", {
  expect_error(
    predict(
      model,
      newdata = conditional_holdout,
      obsID = "obsID",
      panelID = "id",
      conditional = TRUE
    ),
    "only supported for mixed logit models"
  )

  expect_error(
    predict(
      mixed_model_no_panel,
      newdata = conditional_holdout,
      obsID = "obsID",
      panelID = "id",
      conditional = TRUE
    ),
    "only supported for panel mixed logit models"
  )
})
