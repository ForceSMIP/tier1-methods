make_past <- function(data, maxlag = 1, lags = NULL, present = FALSE, ...){
  stopifnot(maxlag>=0)
  stopifnot(all(lags <= maxlag))
  
  if (is.null(data)) return(NULL)
  if (maxlag == 0){
    if (present){
      return(data)
    } else {
      return(NULL)
    }
  }
  if (is.null(lags)) lags <- 1:maxlag
  if (all(lags == 0)){
    if (present){
      return(get_present(data, maxlag = maxlag))
    } else {
      return(NULL)
    }
  }
  n <- nrow(data)
  pasts <- lapply((maxlag:1)[lags], \(lag) data[lag:(n - maxlag + lag - 1),])
  names(pasts) <- paste0("lag", lags)
  pasts$deparse.level = 0
  do.call(cbind, pasts)
}

get_present <- function(data, maxlag = 1){
  data[(maxlag+1):nrow(data), ,drop=FALSE]
}

gpca <- function(X, Y, Z = NULL, maxlag = 1, ...){
  
  UseMethod("gpca", X)
}

gpca.prcomp <- function(X, Y, Z = NULL, maxlag = 1,  ...){
  
  res <- gpca(X$x, Y, Z, maxlag, ...)
  res$grot <- res$rotation
  res$rotation <- X$rotation %*% res$rotation
  return(res)
}


gpca.princomp <- function(X, Y, Z = NULL, maxlag = 1,  ...){
  
  res <- gpca(X$scores, Y, Z = Z, maxlag, ...)
  res$grot <- res$rotation
  res$rotation <- X$loadings %*% res$rotation
  return(res)
}

gpca.default <- function(X, Y, Z = NULL, maxlag = 1, lags = NULL,
                         center = FALSE, scale. = FALSE, 
                         tol = NULL, rank. = NULL, intercept = FALSE){
  
  if (!is.null(Y) && is.null(dim(Y))) Y <- matrix(data = Y, nrow = length(Y))
  if (!is.null(Z) && is.null(dim(Z))) Z <- matrix(data = Z, nrow = length(Z))
  
  d <- ncol(X)
  X <- scale(X, center = center, scale = scale.)
  cen <- attr(X, "scaled:center")
  sc <- attr(X, "scaled:scale")
  Xpast <- make_past(X, maxlag = maxlag, lags = lags$X)
  Zpast <- make_past(Z, maxlag = maxlag,
                     lags = lags$Z, present =  TRUE)##if maxlag = 0 get Z
  control <- cbind(Xpast, Zpast)
  Ypast <- make_past(Y, maxlag = maxlag,
                     lags = lags$Y, present =  TRUE) ## if maxlag = 0 get Y
  if (intercept) control <- cbind(control, rep(1, nrow(Ypast)))
  controlY <- cbind(control, Ypast)
  if (ncol(controlY) < nrow(controlY)){
    qrres <- qr(controlY)
    nc <- ncol(control)
    if (is.null(nc)) nc <- 0
    Q2 <- qr.Q(qrres)[,(nc + 1):ncol(controlY)]    
  }else{
    stop('number of observations in X, minus maxlag, should be greater 
         then number of columns of X times maxlag, try applying PCA and retain 
        just the first components.') 
  }
  
  X0 <- get_present(X, maxlag)
  
  X2 <- t(Q2) %*% X0
  n <- nrow(X2)
  p <- ncol(X2)
  k <- if (!is.null(rank.)) {
    stopifnot(length(rank.) == 1, is.finite(rank.), as.integer(rank.) > 
                0)
    min(as.integer(rank.), n, p)
  }else min(n, p)

  
  res <- svd(X2, nu = 0, nv = k)
  rotation <- res$v
  
  CC <- X %*% rotation
  dimnames(rotation) <- list(colnames(X0), paste0("GPC", seq_len(k)))
  out <- list(x = CC, 
              rotation = rotation, 
              center =  cen, 
              scale = sc,
              model = list(
                x = X0,
                xpast = Xpast,
                zpast = Zpast,
                control = control,
                y = Ypast
              ))
  
  class(out) <- "gpca"
  return(out)
}



make_random_features <- function(x, n = 100, sd = 1){
  b <- 2*pi*runif(1)
  w <- matrix(rnorm(n * ncol(x), mean = 0, sd = sd), ncol = n)
  z <- cos(x %*% w + b) ## eq (15) in https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/
  return(z)
}

make_fourier_features <- function(x, n = 100, b = 1, period = 1){
  w <- matrix(2*pi*(1:n)/period, ncol = n)
  z <- cos(x %*% w + b) ## eq (15) in https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/
  return(z)
}

# C must include B to be a granger test
lm_test <- function(x, y, control = NULL){
  if (is.null(control)){
    model <- lm(x ~ y)
    anova(model, test = "F")$`Pr(>F)`[[1]]
  }else{
    model1 <- lm(x ~ y + control)
    model0 <- lm(x ~ control)
    #lmtest::waldtest(model1, model0)
    anova(model0, model1, test = "F")$`Pr(>F)`[[2]]
  }

}
