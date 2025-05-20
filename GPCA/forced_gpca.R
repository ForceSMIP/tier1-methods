library("terra")
library("zoo")
library("optparse")
source("gpca.R")
source("util.R")
library("cli")
library("glue")
library("glmnet")

option_list = list(
  make_option(c("--out"), type="character", default="output/", 
              help="output file [default= %default]", metavar="character"),
  make_option(c("--input"), type="character", default="data/Evaluation-Tier1/", 
              help="input file [default= %default]", metavar="character"),
  make_option(c("--psl"), type="character", default=NULL, 
              help="input psl file [default= %default]", metavar="character"),
  make_option(c("--forcings"), type="character", default="data/interpolatedForcing.csv", 
              help="file with forcings [default= %default]", metavar="character"),
  make_option(c("--pcarank"), type="integer", default=as.integer(20), 
              help="rank of PCA [default= %default]", metavar="integer"),
  make_option(c("--pcarankpsl"), type="integer", default=as.integer(4), 
              help="rank of PCA for psl [default= %default]", metavar="integer"),
  make_option(c("--degree"), type="integer", default=as.integer(5), 
              help="degreee of poly expansion or number of random basis expansion [default= %default]", metavar="integer"),
  make_option(c("--expansion"), type="character", default="random", 
              help="type of expansion"),
  make_option(c("--period"), type="integer", default=as.integer(5*12), 
              help="period for the fourier expansion [default= %default]", metavar="integer"),
  make_option(c("--maxlag"), type="integer", default=as.integer(5*12), 
              help="general maxlag parameter for GPCA [default= %default]", metavar="integer"),
  make_option(c("--automaxlag"), type="integer", default=as.integer(1), 
              help="autoregressive maxlag parameter for GPCA [default= %default]", metavar="integer"),
  make_option(c("--zmaxlag"), type="integer", default=as.integer(1), 
              help="additional control maxlag parameter for GPCA [default= %default]", metavar="integer"),
  make_option(c("--level"), default=-1, 
              help="significance level to extract components (if negative the absolute value of component are used) [default= %default]"),
  make_option(c("--modelf"), type="character", default="glmnet",
              help="model for the forced direct component(s) lm or glmnet (for lasso, ridge or elasticnet) [default= %default]"),
  make_option(c("--modelnf"), type="character", default="lm",
              help="model for the non-forced component(s) lm or glmnet [default= %default]"),
  make_option(c("--glmnet.alpha"), default = 0.5),
  make_option(c("--scale"),action = "store_true", default = FALSE),
  make_option(c("--center"),action = "store_true", default = FALSE),
  make_option(c("--overwrite"), action = "store_true", default = FALSE)
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

forcing <- read.csv(opt$forcings)

if (dir.exists(opt$input)){
  torun <- list.files(opt$input, pattern="*.nc", recursive = TRUE, full.names = TRUE)
}else{ ## we assume it is a file 
  torun <- opt$input
}


bar <- cli_progress_bar(name = "GPCA forced component",
                        total = length(torun), type = "iterator")

for (finput in torun){
  cli_progress_output("using input file {finput}")
  
  VARIABLE <- strsplit(basename(finput),split =  c("[_.]"))[[1]][1]
  GROUPNAME <- "UV-ISP"
  MEMBERID <- strsplit(basename(finput),split =  c("[_.]"))[[1]][3]
  TIER <- sub("Tier", "", grep("Tier", strsplit(finput, "[-/]")[[1]], value = TRUE))
  METHOD <- "gpca"
  
  if (opt$maxlag == 0){
    METHOD <- "direff"
  }
  
  if (!is.null(opt$psl)){
    METHOD <- glue("{METHOD}-da")
    if (dir.exists(opt$psl)){
      ## match the input to the corresponding psl 
      psl_file <- file.path(opt$psl, sub("day", "mon", sub(VARIABLE, "psl", basename(finput))))
    }else{
      psl_file <- opt$psl
    }
  }else {
    psl_file <- NULL
  }
  
  if (length(TIER) > 0){
    ## we are working with a evaluation file
    outfile <- file.path(opt$out, glue("{VARIABLE}_{MEMBERID}_{TIER}_{METHOD}_{GROUPNAME}.nc"))                  
  } else {
    ## it is a training file
    outfile <- file.path(opt$out, glue("{METHOD}_{basename(finput)}"))   
  }
  
  if (file.exists(outfile)){
    if (!opt$overwrite){
      cli_progress_output(glue("file {finput} skipped as {outfile} exists, run with --overwirte otherwise"))
      cli_progress_update()
      next
    }
  }
  
  x <- rast(finput)
  Y <- forcing[as.yearmon(forcing[,1]) %in% as.yearmon(time(x)), 2]
  ## get climatology 
  x_clim <- app(x, get_clim)
  
  ## get anomalies
  x_an <- x - x_clim
  
  X <- t(values(x_an))
  maskid <- colSums(is.na(X)) == 0
  X <- X[,maskid]
  
  pcX <- prcomp(X, rank. = opt$pcarank, scale. = opt$scale, center = opt$center) ##pca
  
  YY <- matrix(Y, dimnames = list(NULL, "Y"))
  if (opt$expansion == "polynomial"){
    YY <- poly(YY, degree = opt$degree)
  }
  if (opt$expansion == "random"){
    YY <- cbind(YY, make_random_features(YY, opt$degree,  sd = 1))
  }
  if (opt$expansion == "fourier"){
    YY <- cbind(YY, make_fourier_features(YY, opt$degree, period = opt$period))
  }
  
  #
  #
  
  if (!is.null(psl_file)){
    cli_progress_output("using psl_file {psl_file}")
    psl <- rast(psl_file)
    psl_anom <- psl - app(psl, get_clim)
    pslpc <- prcomp(t(values(psl_anom)), rank = opt$pcarankpsl, scale. = opt$scale,
                    center = opt$center)
    Z <- pslpc$x[,-1,drop=FALSE] ## we skip the first pca of psl
  }else{
    Z = NULL
  }
  
  ### cant' use different lag for x  for now see TODO later on
  stopifnot(opt$automaxlag <= 1)
  ####
  
  
  if (opt$automaxlag == 0){
    xlags <- 0
  }else{
    xlags <- 1:opt$automaxlag
  }
  
  ## get only year-lags for the forcings 
  ylags <- c(12 * (1:(opt$maxlag %/% 12)))
  
  if (opt$zmaxlag == 0){
    zlags <- 0
  }else{
    zlags <- 1:opt$zmaxlag
  }
  
  res <- gpca(pcX, Y = YY, Z = Z,
              maxlag = opt$maxlag,
              lags  = list(X = xlags, Z = zlags, Y = ylags), 
              scale. = FALSE,
              center = FALSE,
              intercept = TRUE)
  
  
  
  #### now need to reconstruct the signal 
  present <- get_present(res$x, maxlag = opt$maxlag) 
  
  ## compute granger test pvalues
  pvals <- apply(present, MARGIN = 2,
                 function(xx) lm_test(xx, y = res$model$y, control = res$model$control))
  if (opt$level > 0){
    idx <- which(pvals < opt$level)
  }else{
    idx <- sort(pvals, decreasing = FALSE, index.return = TRUE)$ix[1:as.integer(abs(opt$level))]
  }
  if (length(idx) == 0) idx <- 1
  
  cli_progress_output("used granger components at level {opt$level}: {glue_collapse(idx, sep = ', ')}")
  if (opt$modelf == "glmnet"){
    cARF <- sapply(idx, function(idxi){
      modelARF <- cv.glmnet(x = cbind(res$model$control, res$model$y), y = present[,idxi],
                            alpha = opt$glmnet.alpha, intercept = FALSE,
                            family="gaussian") 
      as.matrix(coef(modelARF, s = "lambda.min"))[-1,,drop = FALSE]
    })
  }
  if (opt$modelf == "lm"){
    modelARF <- lm.fit(x = cbind(res$model$control, res$model$y),
                       y = present[,idx,drop = FALSE]) 
    cARF <- coef(modelARF)
    cARF[is.na(cARF)] <- 0
  }
  ## split coefficients 
  cARF.y <- tail(cARF, ncol(res$model$y))
  cARF.c <- head(cARF, ncol(res$model$control))
  

  if (opt$modelnf == "glmnet"){
    modelAR  <- cv.glmnet(x = res$model$control,
                          y = present[,-idx, drop = FALSE],
                          family = "mgaussian", alpha = opt$glmnet.alpha,
                          intercept = FALSE)
    cAR <- sapply(coef(modelAR, s = "lambda.min"),
                  function(ccc) as.matrix(ccc))[-1,, drop = FALSE]
  }
  if (opt$modelnf == "lm"){
    modelAR  <- lm.fit(x = res$model$control, y = present[,-idx, drop = FALSE])
    cAR <- coef(modelAR)
    cAR[is.na(cAR)] <- 0
  }
  
  newcontrol <- rbind(matrix(colMeans(res$model$control[1:opt$maxlag,]),
                             nrow = opt$maxlag, ncol = ncol(res$model$control), byrow = TRUE), 
                      res$model$control)
  newcontrol[,ncol(newcontrol)] <- 1 #fix intercept
  
  
  ### fix it, maybe it works now
  ## here we fix the psl-iv to 0
  if (!is.null(Z)){
    if (!is.null(res$model$xpast)){
      newcontrol[,(ncol(res$model$xpast)+1):(ncol(res$model$xpast)+ncol(res$model$zpast))] <- 0
    }else{
      newcontrol[,1:ncol(res$model$zpast)] <- 0
    }
  }
  
  new <- res$x
  new[1:opt$maxlag,] <- colMeans(res$x[1:opt$maxlag,])
  
  ## pad it with 0s
  ## TODO: we could use the forcing time series from the past
  initY <- matrix(colMeans(res$model$y[1:opt$maxlag,]),
                  nrow = opt$maxlag, ncol = ncol(res$model$y), byrow = TRUE)
  Y0 <- rbind(initY,
              res$model$y)

  
  for (i in 1:nrow(new)){
    new[i, idx] <-  newcontrol[i, , drop=FALSE] %*% cARF.c + Y0[i,,drop=FALSE] %*% cARF.y 
    new[i, -idx] <- newcontrol[i, , drop=FALSE] %*% cAR
    if (all(xlags > 0)){
      ## back to the pca space
      pcrec <- new[i,] %*% t(res$grot)
      ## TODO: fix it now works only for xlag== 1
      if (i < nrow(newcontrol)) newcontrol[i + 1, 1:ncol(res$model$xpast)] <- pcrec
    }
  }
# 
  
  fr_est <- new %*% t(res$rotation)
  if (opt$scale){ ## scale back data
    fr_est <- fr_est %*% diag(pcX$scale)
  }
  if (opt$center){ ## add center again
    fr_est <- fr_est + matrix(pcX$center, byrow = TRUE, 
                              ncol = length(pcX$center), 
                              nrow = nrow(forcedcomponent))
  }

  ## we now refill the columns with NA
  final_forced <- array(dim = c(nrow(fr_est), length(maskid)))
  final_forced[, maskid] <- fr_est
  ## the rest will be NA
  
  ## the output should have same dimensions as the input
  output <- rast(nrows = nrow(x), ncols = ncol(x), nlyrs = nlyr(x),
                 crs = crs(x), extent = ext(x), time = time(x))
  values(output) <- t(final_forced)
  units(output) <- units(x)
  
  dir.create(opt$out, showWarnings = FALSE, recursive = TRUE)
  
  atts <- paste0(names(opt), "=", opt)
  atts <- c(atts, paste0("VARIABLE=", VARIABLE),
            paste0("METHOD=", METHOD) )
  writeCDF(output, filename = outfile, overwrite = TRUE, 
           atts = paste0(names(opt), "=", opt))
  cli_progress_update()
}

