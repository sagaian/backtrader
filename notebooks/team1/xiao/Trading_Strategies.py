
import sys
import os.path
sys.path.append('C:/Users/xiaog/Dropbox/Mathematics_Finance/Capstone Investment Projects/program development')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy
import utils as utl
plt.style.use("ggplot")

"""
This file contains all the trading strategies that are used to generate the trades
At present, it includes the trading strategies based on the market regimes. It first
receives the data frame and market regimes data, then it determines how to trade based
on these regimes

"""

class Macro_Trading_Session():

    def __init__(self, datadict, market_regime, symbols, strategy):
        """
        Initialize the class
        :param datadict: The data dictionary contains all the tickers that are possible to be traded.
                         datadict[ticker] will return the Yahoo daily typed data of that ticker
        :param market_regime: pandas dataframe with time index and one column of market regime status (0, 1),
                              either momentum or mean-reversion. The column name should be "state"
        :param symbols: a list contains all the tickers that have data in datadict
        :param strategy: a string indicating which strategy is to be used
                         (currently support "moving_average", "mean_reverting")
        """
        self.datadict = copy.copy(datadict)
        self.market_regime = market_regime
        self.symbols = symbols
        self.strategy = strategy

        # indicate the trading action, "long", "short", "momentum", "mean_reverting", "inactivate"
        self.trading_action = "inactivate"

        # for certain strategies
        self.lookbackwindow = 0

        # for moving average
        self.shortwindow = 0

        # parameters for mean_reverting strategy, which characterize the sharpe ratio
        self.high_thresh_0 = 1.25
        self.high_thresh_1 = 0.75
        self.low_thresh_0 = -1.25
        self.low_thresh_1 = -0.5

        # parameters for mean reverting strategy, which specify the number of
        # stocks we choose in the upper return and lower return ranges
        self.upper_num = 0
        self.lower_num = 0
        self.switch = "on"



    def obtain_datadict(self):
        """

        :return: the data dictionary
        """
        return self.datadict

    def set_window_moving_average(self, longwindow, shortwindow):
        """
        This function sets the look back windows for the moving average strategy
        :param longwindow: int
        :param shortwindow: int
        :return: None
        """
        if longwindow <= shortwindow:
            print("The longer lookback window should be larger:")
            return
        else:
            self.lookbackwindow = longwindow
            self.shortwindow = shortwindow

    def update_parameters_mean_reverting(self, longwindow, shortwindow,
                                         lower_num, upper_num, switch="on"):
        """
        This function updates the parameters for mean_reverting strategy
        :param longwindow: int variable, which specify the lookback window of the data for momentum status
        :param shortwindow: int variable, which specify the lookback window of the data for mean reverting status
        :param lower_num: int variable, which specify the number of stocks with lowest returns in the trading
        :param upper_num: int variable, which specify the number of stocks with largest returns in the trading
        :param switch: specify if we switch from momentum to mean_reverting
        :return:
        """
        self.lookbackwindow = longwindow
        self.shortwindow = shortwindow
        self.upper_num = upper_num
        self.lower_num = lower_num
        self.switch = switch
        return


    def update_lookbackwindow(self, lookbackwindow):
        """
        This function updates the lookback window for other strategies
        :param lookbackwindow: int
        :return: None
        """
        self.lookbackwindow = lookbackwindow


    def add_data(self, dataframe, symbol):
        """
        This function add data to the original datadict
        :param dataframe: pandas dataframe (Yahoo typed daily bars)
        :param symbol: stock/equity ticker
        :return: None
        """
        self.datadict[symbol] = dataframe
        return

    def update_market_regime(self, market_regime):
        """
        This function updates the market regime. It should be called before trading begins
        :param market_regime: the pandas dataframe contains new market regime
        :return: None
        """
        self.market_regime = market_regime

    def begin_trade(self, tickers, begin_date, end_date, initial_equity=1000000.0, leverage=2):
        """
        This function will be called once we start to trade.
        :param tickers: a list that contains tickers that will be traded.
        :param initial_equity: initial amount
        :param begin_date: the beginning trading date (string of type %Y-%m-%d)
        :param end_date : the ending trading date (string of type %Y-%m-%d)
        :param leverage: the maximum leverage we can use. i.e. leverage =2 means we can borrow
        capital equals to the initial capital. It also means the short amount should be no greater than
        the current cash values
        :return: The dollar positions of our tradings along the time
        """
        traded_datadict = dict()
        for ticker in tickers:
            traded_datadict[ticker] = self.datadict[ticker]

        begin_date = pd.Timestamp(np.datetime64(begin_date))
        end_date = pd.Timestamp(np.datetime64(end_date))

        # check if the begin_date and end_date are legitimate
        for ticker, data in traded_datadict.items():
            # check the initial time of the dataframe is in the range of trading dates
            if data.index[0] > begin_date or data.index[-1] < end_date:
                print("The range of trading dates range exceeds the range of recorded stock information")
                return

        # return the number of tickers in the list
        ticker_num = len(tickers)

        # create the P&L series for the trading periods.
        trading_summary = dict()

        if self.strategy == "moving_average":

            for ticker in tickers:
                ticker_equity = initial_equity/ticker_num
                print("start trading")
                trading_summary[ticker] = self.moving_average_strategy(ticker,
                                                                       traded_datadict[ticker],
                                                                        begin_date,
                                                                        end_date,
                                                                        ticker_equity)
        elif self.strategy == "mean_reverting":

            ETF_adjusted_closed = utl.obtain_adjusted_close(self.datadict)
            trading_summary = self.mean_reverting_strategy(
                ETF_adjusted_closed,
                begin_date,
                end_date,
                initial_equity,
                leverage
            )

        return trading_summary



    def moving_average_strategy(self, ticker, tickerdata, begin_date, end_date, initial_equity, leverage=2):
        """
        This function performs the long_biased moving average strategy for single ticker.
        :param tickerdata: the pandas dataframe for the ticker (contains the time index and adj close price)
        :param begin_date: trading begin date (pandas timestamp)
        :param end_date: trading end date (pandas timestamp)
        :param initial_equity: float
        :param leverage: the maximum leverage we can use. i.e. leverage =2 means we can borrow
        capital equals to the initial capital. It also means the short amount should be no greater than
        the current cash values
        :return: the dollar position of our investment
        """

        trading_action = [] # record trading_action, "buy" ,"sell", "waiting"
        dollar_positions = []

        # Initialize our investment account
        equity_value = 0.0
        equity_quantity = 0
        cash_value = initial_equity


        # The following code retrieve the index in between begin_date and end_date
        # and it is in the legal data range of the ticker dataset

        # trading time
        time_index = tickerdata[begin_date:end_date].index

        # begin the trading process
        for index, time_point in enumerate(time_index):
            # obtain the previous time
            previous_time_point = pd.Timestamp(np.busday_offset(time_point, -1, roll="forward"))
            # obtain the longer lookback window time point
            long_window_time = pd.Timestamp(np.busday_offset(previous_time_point,
                                                             -self.lookbackwindow, roll="forward"))

            # obtain the shorter lookback window time point
            short_window_time = pd.Timestamp(np.busday_offset(previous_time_point,
                                                             -self.shortwindow, roll="forward"))

            # If the longer lookback window time point is out of the range of time index of
            # ticker data, we skip this iteration
            if long_window_time <= tickerdata.index[0]:
                dollar_positions.append(cash_value)
                continue

            # calculate the moving average mean
            short_sma = tickerdata.loc[short_window_time:previous_time_point]["Adj Close"].mean()
            long_sma = tickerdata.loc[long_window_time:previous_time_point]["Adj Close"].mean()


            # initialize the trading action at this time
            trading_action_this_time = "waiting"


            if previous_time_point in self.market_regime.index:

                if self.trading_action == "inactivate":
                    # we currently purchase the ticker once the short moving average is dominating
                    if short_sma > long_sma and self.market_regime.loc[previous_time_point]["state"] == 0:

                        print("LONG %s at %s" % (ticker, time_point))
                        equity_quantity = math.floor(cash_value/tickerdata.loc[time_point]["Adj Close"])
                        equity_value = equity_quantity * tickerdata.loc[time_point]["Adj Close"]
                        cash_value = cash_value - equity_value
                        self.trading_action = "long"
                        trading_action_this_time = "buy"


                elif self.trading_action == "long":

                    if self.market_regime.loc[previous_time_point]["state"] == 1:

                        print("SHORT %s at %s" % (ticker, time_point))
                        cash_value = cash_value + equity_quantity * tickerdata.loc[time_point]["Adj Close"]
                        equity_value = 0.0
                        equity_quantity = 0
                        self.trading_action = "inactivate"
                        trading_action_this_time = "sell"

                    elif self.market_regime.loc[previous_time_point]["state"] == 0 and short_sma < long_sma:

                        print("SHORT %s at %s" % (ticker, time_point))
                        cash_value = cash_value + equity_quantity * tickerdata.loc[time_point]["Adj Close"]
                        equity_value = 0.0
                        equity_quantity = 0
                        self.trading_action = "inactivate"
                        trading_action_this_time = "sell"

            trading_action.append(trading_action_this_time)
            # compute equity value for this trade
            equity_value = equity_quantity * tickerdata.loc[time_point]["Adj Close"]
            dollar_positions.append(cash_value + equity_value)


        trading_summary = pd.DataFrame(np.c_[trading_action, dollar_positions],
                                       index=time_index, columns=["trading actions", "dollar positions"])
        return trading_summary


    def mean_reverting_strategy(self, adjusted_closed_data, begin_date, end_date, initial_equity, leverage):
        """
        This strategy

        :param adjusted_closed_data: dataframe contains all the adjusted closed prices of the data
        :param start_date:
        :param end_date:
        :param initial_equity:
        :param leverage:
        :return:
        """
        # list initialization
        # store the positions of the stocks
        short_position = np.zeros(self.lower_num)  # short positions
        long_position = np.zeros(self.upper_num) # long positions


        trading_actions = []
        dollar_positions = []  # dollar positions
        regimes_hist = []  # regime indicators

        # specify the dollar positions
        long_equity_value = 0.0
        short_equity_value = 0.0
        cash_value = initial_equity

        # This indicates that if one of the regime lasts for a very long time, we have to rebalance
        # the portfolio
        rebalance_indicator = 0

        # obtain the trading periods
        time_index = adjusted_closed_data[begin_date:end_date].index

        for index, time_point in enumerate(time_index):
            # obtain the previous time
            previous_time_point = pd.Timestamp(np.busday_offset(time_point, -1, roll="forward"))
            # obtain the longer lookback window time point
            long_window_time = pd.Timestamp(np.busday_offset(previous_time_point,
                                                             -self.lookbackwindow, roll="forward"))

            # obtain the shorter lookback window time point
            short_window_time = pd.Timestamp(np.busday_offset(previous_time_point,
                                                             -self.shortwindow, roll="forward"))
            # If there are no enoough data, we just skip this
            if long_window_time <= adjusted_closed_data.index[0]:
                trading_actions.append("waiting")
                dollar_positions.append(cash_value)
                continue

            # specify the beginning trading actions
            trading_actions_this_time = "waiting"

            # judge if we receive the signal
            if previous_time_point in self.market_regime.index:
                # we first initialize the trade
                if self.trading_action == "inactivate":
                    # if we encounter momentum status
                    if self.market_regime.loc[previous_time_point]["state"] == 0:
                        # Now we should obtain the upper_num stocks with highest returns and long them
                        # We should obtain the lower_num stocks with the lowest returns and short them

                        # retrieve the data and tickers for high and low returns
                        data_temp = adjusted_closed_data.loc[long_window_time:previous_time_point]
                        ticker_high, ticker_low = utl.obtain_high_low_return_equity_symbol(data_temp,
                                                                                       self.upper_num,
                                                                                       self.lower_num)

                        # Now we execute the trade
                        # First of all, we obtain the price series
                        price_vec_high = adjusted_closed_data.loc[time_point][ticker_high]
                        price_vec_low = adjusted_closed_data.loc[time_point][ticker_low]


                        long_position, short_position, cash_value = utl.obtain_trading_positions(
                            price_vec_high,
                            price_vec_low,
                            cash_value,
                            leverage
                        )

                        rebalance_indicator = 0
                        self.trading_action = "momentum"
                        trading_actions_this_time = "momentum long/short"

                    elif self.market_regime.loc[previous_time_point]["state"] == 1 and self.switch == "on":
                        # Now we should obtain the upper_num stocks with highest returns and short them
                        # We should obtain the lower_num stocks with the lowest returns and long them

                        data_temp = adjusted_closed_data.loc[short_window_time:previous_time_point]
                        ticker_high, ticker_low = utl.obtain_high_low_return_equity_symbol(data_temp,
                                                                                           self.upper_num,
                                                                                           self.lower_num)

                        # Now we execute the trade
                        # First of all, we obtain the price series
                        price_vec_high = adjusted_closed_data.loc[time_point][ticker_high]
                        price_vec_low = adjusted_closed_data.loc[time_point][ticker_low]

                        long_position, short_position, cash_value = utl.obtain_trading_positions(
                            price_vec_low,
                            price_vec_high,
                            cash_value,
                            leverage
                        )
                        rebalance_indicator = 0
                        self.trading_action = "mean_reverting"
                        trading_actions_this_time = "mean_reverting long/short"

                # should we switch from momentum to mean reverting
                elif (self.trading_action == "momentum" and
                     self.market_regime.loc[previous_time_point]["state"] == 1):
                    # we need to decide if we switch from momentum to mean reverting
                    if self.switch == "on":
                        # we first clear the long short positions
                        net_cash_value = long_equity_value + cash_value - short_equity_value

                        data_temp = adjusted_closed_data.loc[short_window_time:previous_time_point]
                        ticker_high, ticker_low = utl.obtain_high_low_return_equity_symbol(data_temp,
                                                                                           self.upper_num,
                                                                                           self.lower_num)

                        price_vec_high = adjusted_closed_data.loc[time_point][ticker_high]
                        price_vec_low = adjusted_closed_data.loc[time_point][ticker_low]

                        long_position, short_position, cash_value = utl.obtain_trading_positions(
                            price_vec_low,
                            price_vec_high,
                            net_cash_value,
                            leverage
                        )

                        rebalance_indicator = 0
                        self.trading_action = "mean_reverting"
                        trading_actions_this_time = "mean_reverting long/short"
                    elif self.switch == "off":
                        cash_value = long_equity_value + cash_value - short_equity_value
                        self.trading_action = "inactivate"
                        trading_actions_this_time = "close"

                elif (self.trading_action == "mean_reverting" and
                          self.market_regime.loc[previous_time_point]["state"] == 0):
                    # we need to switch the strategy from mean_reverting to momentum

                    # we first clear the long short positions
                    net_cash_value = long_equity_value + cash_value - short_equity_value

                    data_temp = adjusted_closed_data.loc[long_window_time:previous_time_point]
                    ticker_high, ticker_low = utl.obtain_high_low_return_equity_symbol(data_temp,
                                                                                       self.upper_num,
                                                                                       self.lower_num)

                    price_vec_high = adjusted_closed_data.loc[time_point][ticker_high]
                    price_vec_low = adjusted_closed_data.loc[time_point][ticker_low]

                    long_position, short_position, cash_value = utl.obtain_trading_positions(
                        price_vec_high,
                        price_vec_low,
                        net_cash_value,
                        leverage
                    )

                    rebalance_indicator = 0
                    self.trading_action = "momentum"
                    trading_actions_this_time = "momentum long/short"


                elif rebalance_indicator >= self.shortwindow:
                    # we need to rebalance
                    if self.trading_action == "momentum":
                        # rebalance according to the momentum strategy
                        # retrieve the data and tickers for high and low returns
                        data_temp = adjusted_closed_data.loc[long_window_time:previous_time_point]
                        ticker_high, ticker_low = utl.obtain_high_low_return_equity_symbol(data_temp,
                                                                                           self.upper_num,
                                                                                           self.lower_num)

                        # Now we execute the trade
                        # First of all, we obtain the price series
                        price_vec_high = adjusted_closed_data.loc[time_point][ticker_high]
                        price_vec_low = adjusted_closed_data.loc[time_point][ticker_low]

                        net_cash_value = long_equity_value + cash_value - short_equity_value

                        long_position, short_position, cash_value = utl.obtain_trading_positions(
                            price_vec_high,
                            price_vec_low,
                            net_cash_value,
                            leverage
                        )

                        rebalance_indicator = 0
                        self.trading_action = "momentum"
                        trading_actions_this_time = "momentum long/short"

                    elif self.trading_action == "mean_reverting":
                        # rebalance according to the mean reverting strategy
                        # We should obtain the lower_num stocks with the lowest returns and long them

                        data_temp = adjusted_closed_data.loc[short_window_time:previous_time_point]
                        ticker_high, ticker_low = utl.obtain_high_low_return_equity_symbol(data_temp,
                                                                                           self.upper_num,
                                                                                           self.lower_num)

                        # Now we execute the trade
                        # First of all, we obtain the price series
                        price_vec_high = adjusted_closed_data.loc[time_point][ticker_high]
                        price_vec_low = adjusted_closed_data.loc[time_point][ticker_low]

                        net_cash_value = long_equity_value + cash_value - short_equity_value

                        long_position, short_position, cash_value = utl.obtain_trading_positions(
                            price_vec_low,
                            price_vec_high,
                            net_cash_value,
                            leverage
                        )
                        rebalance_indicator = 0
                        self.trading_action = "mean_reverting"
                        trading_actions_this_time = "mean_reverting long/short"
                else:
                    # we do not have to do nothing
                    trading_actions_this_time = "waiting"

            rebalance_indicator += 1
            # retrieve the prices of long and short positions
            if self.trading_action != "inactivate":
                price_long = adjusted_closed_data.loc[time_point][long_position.index]
                price_short = adjusted_closed_data.loc[time_point][short_position.index]

                # compute the long position and short position values
                long_equity_value = long_position.dot(price_long)
                short_equity_value = short_position.dot(price_short)

                # compute the total values
                total_value = cash_value + long_equity_value - short_equity_value

                # append the values
                dollar_positions.append(total_value)
                trading_actions.append(trading_actions_this_time)
            else:
                dollar_positions.append(cash_value)
                trading_actions.append(trading_actions_this_time)



        trading_summary = pd.DataFrame(np.c_[trading_actions, dollar_positions],
                                       index=time_index, columns=["trading actions", "dollar positions"])
        return trading_summary






























