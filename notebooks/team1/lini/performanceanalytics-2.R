library(PerformanceAnalytics)
library(quantmod)

# read CSV
hmm_switch = read.csv("hmm_regime_wealth_discontinuous.csv",header=TRUE)
head(hmm_switch)
series = hmm_switch["Date"]
time_stamp = as.POSIXct(series[2:nrow(series),1],format="%Y-%M-%D")



trade_data = as.numeric(unlist(hmm_switch["mr_wealth"]))
dates <- as.Date(hmm_switch$Date)
cum_pnl <- xts(hmm_switch['mr_wealth'], order.by = dates)
daily_ret <- Delt(cum_pnl)
SharpeRatio.annualized(daily_ret)


trade_data = as.numeric(unlist(hmm_switch["mo_wealth"]))
dates <- as.Date(hmm_switch$Date)
cum_pnl <- xts(hmm_switch['mo_wealth'], order.by = dates)
daily_ret <- Delt(cum_pnl)
SharpeRatio.annualized(daily_ret)

trade_data = as.numeric(unlist(hmm_switch["mr_wealth_cross"]))
dates <- as.Date(hmm_switch$Date)
cum_pnl <- xts(hmm_switch['mr_wealth_cross'], order.by = dates)
daily_ret <- Delt(cum_pnl)
SharpeRatio.annualized(daily_ret)


trade_data = as.numeric(unlist(hmm_switch["mo_wealth_cross"]))
dates <- as.Date(hmm_switch$Date)
cum_pnl <- xts(hmm_switch['mo_wealth_cross'], order.by = dates)
daily_ret <- Delt(cum_pnl)
SharpeRatio.annualized(daily_ret)
