library(zoo)

## read forcings
forcing <- read.csv("data/forcings.csv")

## make time series monthly
Y_ts <- ts(forcing[forcing$year %in% 1880:2022, "total"], start = 1880,
           end = tail(forcing$year, 1), frequency = 1)

YY <- ts(NA, start = c(1880,1), end = c(2022,12), frequency = 12)
YY[(0:139)*12 + 1] <- Y_ts
YY <- zoo::na.approx(YY, method = "linear", na.rm = FALSE)

## now we extrapolate
model <- lm(YY ~ time, data = data.frame(YY = YY[1640:1716], time = time(YY)[1640:1716]))
YY[is.na(YY)] = predict(model, newdata = data.frame(YY = NA, time = time(YY)[is.na(YY)]))


write.csv(data.frame(time = as.Date(YY) + 15 , totalforcing= YY), file = "data/interpolatedForcing.csv", row.names = FALSE)
