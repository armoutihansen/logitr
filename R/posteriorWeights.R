# ============================================================================
# Posterior weight engine for conditional prediction
# ============================================================================

getPredictVFunctions <- function(modelSpace) {
  getV <- getMnlV_pref
  getVDraws <- getMxlV_pref
  if (modelSpace == "wtp") {
    getV <- getMnlV_wtp
    getVDraws <- getMxlV_wtp
  }
  list(getV = getV, getVDraws = getVDraws)
}

makePosteriorWeightEngine <- function(object) {
  predictFuncs <- getPredictVFunctions(object$modelSpace)
  panelIDs <- object$data$panelIDValues
  weights <- computePosteriorWeights(object, predictFuncs$getVDraws)
  rownames(weights) <- as.character(panelIDs)
  structure(
    list(panelIDs = panelIDs, weights = weights),
    class = "posterior_weight_engine"
  )
}

computePosteriorWeights <- function(object, getVDraws) {
  data_diff <- makeDiffData(object$data, object$modelType)
  historyDraws <- getPredictionDraws(
    object,
    list(
      X = data_diff$X,
      scalePar = data_diff$scalePar,
      obsID = data_diff$obsID
    ),
    useDiffData = TRUE,
    getVDraws = getVDraws
  )
  logPosterior <- rowsum(
    log(historyDraws),
    group = data_diff$panelID,
    reorder = FALSE
  )
  logPosterior <- sweep(logPosterior, 1, apply(logPosterior, 1, max), "-")
  posterior <- exp(logPosterior)
  posterior / rowSums(posterior)
}

matchPosteriorWeightPanelIDs <- function(engine, panelIDVals, panelIDName) {
  panelIndex <- match(as.character(panelIDVals), as.character(engine$panelIDs))
  if (anyNA(panelIndex)) {
    missing <- unique(panelIDVals[is.na(panelIndex)])
    missing <- paste(missing, collapse = ", ")
    stop(
      'The "panelID" column contains values not seen during model estimation: ',
      missing
    )
  }
  panelIndex
}

getPredictionDraws <- function(object, data, useDiffData = FALSE, getVDraws = NULL) {
  if (is.null(getVDraws)) {
    getVDraws <- getPredictVFunctions(object$modelSpace)$getVDraws
  }
  coefs <- stats::coef(object)
  betaDraws <- makeBetaDraws(
    coefs, object$parIDs, object$n, object$standardDraws, object$inputs$correlation
  )
  colnames(betaDraws) <- names(object$parSetup)
  betaDraws <- selectDraws(betaDraws, object$modelSpace, data$X)
  VDraws <- getVDraws(betaDraws, data$X, data$scalePar, object$n)
  if (useDiffData) {
    return(getLogit(exp(VDraws), data$obsID))
  }
  predictLogit(VDraws, data$obsID)
}
