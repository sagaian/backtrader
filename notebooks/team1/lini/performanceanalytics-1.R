library(PerformanceAnalytics)
library(quantmod)

# read CSV
hmm_switch = read.csv("0416-hmm-ETF-wealth.csv",header=TRUE)
head(hmm_switch)
series = hmm_switch["Date"]
time_stamp = as.POSIXct(series[2:nrow(series),1],format="%m/%d/%Y")



trade_data = as.numeric(unlist(hmm_switch["scored_v1"]))
dates <- as.Date(hmm_switch$Date, format="%m/%d/%Y")
cum_pnl <- xts(hmm_switch['scored_v1'], order.by = dates)
daily_ret <- Delt(cum_pnl)
SharpeRatio.annualized(daily_ret)


trade_data = as.numeric(unlist(hmm_switch["scored_v2"]))
dates <- as.Date(hmm_switch$Date, format="%m/%d/%Y")
cum_pnl <- xts(hmm_switch['scored_v2'], order.by = dates)
daily_ret <- Delt(cum_pnl)
SharpeRatio.annualized(daily_ret)



trade_data = as.numeric(unlist(hmm_switch["wealth_6m"]))
dates <- as.Date(hmm_switch$Date, format="%m/%d/%Y")
cum_pnl <- xts(hmm_switch['wealth_6m'], order.by = dates)
daily_ret <- Delt(cum_pnl)
SharpeRatio.annualized(daily_ret)


trade_data = as.numeric(unlist(hmm_switch["wealth_8m"]))
dates <- as.Date(hmm_switch$Date, format="%m/%d/%Y")
cum_pnl <- xts(hmm_switch['wealth_8m'], order.by = dates)
daily_ret <- Delt(cum_pnl)
SharpeRatio.annualized(daily_ret)


trade_data = as.numeric(unlist(hmm_switch["wealth_10m"]))
dates <- as.Date(hmm_switch$Date, format="%m/%d/%Y")
cum_pnl <- xts(hmm_switch['wealth_10m'], order.by = dates)
daily_ret <- Delt(cum_pnl)
SharpeRatio.annualized(daily_ret)



trade_data = as.numeric(unlist(hmm_switch["wealth_12m"]))
dates <- as.Date(hmm_switch$Date, format="%m/%d/%Y")
cum_pnl <- xts(hmm_switch['wealth_12m'], order.by = dates)
daily_ret <- Delt(cum_pnl)
SharpeRatio.annualized(daily_ret)
