get_clim <- function(x){
  tmp <- matrix(ncol = 12, data = x, byrow = TRUE)
  apply(tmp, 2, mean)
}

ma <- function(x, n = 10){ as.numeric(stats::filter(x, rep(1/n,n), sides = 1))}

