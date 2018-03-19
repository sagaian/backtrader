

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.dates import YearLocator
from matplotlib.dates import MonthLocator
import datetime
import statsmodels.tsa.stattools as ts
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.decomposition import PCA
import statistics as stat



# test function
def f(a, b):
    return a + b


# retrieve the adjusted closed prices from all the ETF tickers
def obtain_adjusted_close(datadict):
    """
    The function retrieve all the adjusted close prices of the datadict
    """
    initial = 0
    keys = list(datadict.keys())
    for key, value in datadict.items():
        if key != "XLRE":
            if initial == 0:
                ETF_adjclose = np.array(value["Adj Close"]).reshape(-1,1)
                initial += 1
            else:
                new_adjclose = np.array(value["Adj Close"]).reshape(-1,1)
                ETF_adjclose = np.column_stack((ETF_adjclose, new_adjclose))
                initial += 1
        else:
            continue
    ETF_adjclose = pd.DataFrame(ETF_adjclose, index=datadict[keys[0]].index, columns=keys)
    return ETF_adjclose


def obtain_high_low_return_equity_symbol(dataframe, upper_num, lower_num):
    """
    This function receives the dataframe with adj close prices as input and rank the historical returns
    It then returns the symbols of highest upper_num symbols of equities and lowest lower_num symbols of
    equities
    :param dataframe: pandas dataframe contains adj closed prices
    :param upper_num: number of highest return stocks
    :param lower_num: number of lowest return stocks
    :return: ticker symbols for highest upper_num return stocks and lowest lower_num return stocks
    """
    mean_vec = dataframe.pct_change().dropna().mean()
    return mean_vec.nlargest(upper_num).index, mean_vec.nsmallest(lower_num).index

def obtain_trading_positions(price_high, price_low, cash_value, leverage):
    """
    This function computes the holding positions for upper_num and lower_num as well as cash values.
    It assumes that we long the high return stocks and short low return stocks. If we would like to do the
    reverse, we can just reverse the symbols
    :param price_high: series (The current prices for the tickers with high return)
    :param price_low: series (The current prices for the tickers with low return)
    :param cash_value: float net cash values
    :param leverage: trading leverage we can use
    :return: trading positions with high and low return stocks as well as cash values
    """
    # now we compute the short equity_value
    # specify the short value
    short_equity_value = cash_value * (leverage - 1)
    long_equity_value = cash_value * leverage

    upper_num = len(price_high)
    lower_num = len(price_low)

    each_equity_value = short_equity_value / lower_num
    short_position = np.floor(np.ones(lower_num) * each_equity_value / price_low)
    short_equity_value = short_position.dot(price_low)

    long_equity_value_orig = cash_value + short_equity_value
    each_equity_value = long_equity_value_orig / upper_num
    long_position = np.floor(np.ones(upper_num) * each_equity_value / price_high)
    long_equity_value = long_position.dot(price_high)
    cash_value = long_equity_value_orig - long_equity_value

    return long_position, short_position, cash_value


def data_preprocessing(main_dir, symbols, sub_dir):
    """
    This function receives main directory and sub directory as well as ticker symbols
    :param main_dir: main directory (string)
    :param symbols: ticker symbols (list)
    :param sub_dir: sub directory (string)
    :return: dictionary consist of the dataframe for all the tickers
    """
    # create data dictionary
    ETF_dict = dict()
    # retreive data for each symbol
    for symbol in symbols:
        # process the data into pandas format
        data = pd.read_csv(main_dir + sub_dir + "/{}.csv".format(symbol), header=0, index_col=0, parse_dates=True)
        ETF_dict[symbol] = data
    return ETF_dict


def PCA_analysis(dataframe_return, days_diff):
    """
    This function receives a pandas dataframe that contains the returns for all the tickers and
    perform the PCA analysis on the return data from (days_diff) back to one day before.
    It returns the explained ratio and singular values stored as a dictionary
    with time as index
    """
    # Initialize the time_index that performs PCA analysis
    time_index = []
    # Initialize the dictionary that contains the explained variance ratio and
    # singular values of the PCA
    ETF_integration = dict()
    # obtain a four weeks time slots

    for time in dataframe_return.index:
        # obtain the last time_index
        last_time = pd.Timestamp(np.busday_offset(time, -days_diff, roll='forward'))

        # make sure the last time index is not out of the range of the data
        if last_time >= dataframe_return.index[0]:
            # retrieve the index between these two time points
            index_between = (dataframe_return.index >= last_time) & (dataframe_return.index <= time)
            # store the data in between
            tmp_data = dataframe_return[index_between]
            # Now we perform the PCA analysis
            pca = PCA()
            pca.fit(tmp_data)
            # obtain the explained ratio and singular values
            ratio, singularValue = pca.explained_variance_ratio_, pca.singular_values_
            time_index.append(time)
            ETF_integration[time] = (ratio, singularValue)

    return time_index, ETF_integration



def create_states(time_index, explained_ratio, percentile_val, time_point):
    """
    This function receives explained_ratio as numpy array and value percentile
    and returns a pandas dataframe with state variable

    """
    time_point = pd.Timestamp(time_point)
    z_value = np.percentile(explained_ratio.loc[:time_point], percentile_val)
    state_column = explained_ratio > z_value
    state_data = pd.DataFrame(np.c_[explained_ratio, state_column], index=time_index,
                              columns=["explained ratio", "state"])
    state_data["state"] = state_data["state"].astype(int)
    return state_data


def obtain_prices_df(csv_filepath, start_date, end_date=None):
    """
    Obtain the prices DataFrame from the CSV file,
    filter by the end date and calculate the
    percentage returns.
    """
    df = pd.read_csv(
        csv_filepath, header=0,
        names=[
            "Date", "Open", "High", "Low",
            "Close",  "Adj Close", "Volume"
        ],
        index_col="Date", parse_dates=True
    )
    df["Returns"] = df["Adj Close"].pct_change()
    start_date = pd.Timestamp(np.datetime64(start_date))
    if not end_date:
        df = df.loc[start_date:]
    else:
        end_date = pd.Timestamp(np.datetime64(end_date))
        df = df.loc[start_date:end_date]
    df.dropna(inplace=True)
    return df

# The following function is used to visualize the movement of data with respect to different states
def plot_sample_hidden_states(states, df, components, ticker):
    """
    Plot the adjusted closing prices masked by
    the in-sample states as a mechanism
    to understand the market regimes.
    """
    # Create the correctly formatted plot
    fig, axs = plt.subplots(
        components,
        sharex=True, sharey=True,
        figsize= (15, 4)
    )
    colours = cm.rainbow(
        np.linspace(0, 1, components)
    )
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = states == i
        ax.plot_date(
            df.index[mask],
            df["Adj Close"][mask],
            ".", linestyle='none',
            c=colour
        )
        ax.set_title("State #%s" % i + " " + ticker)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.grid(True)
    plt.show()


def obtain_sharpe_ratio(dataframe, interest_rate, benchmark=False):
    """
    This function computes sharpe ratio for the trading actions
    """
    if benchmark == False:
        returns = dataframe["dollar positions"].astype(float).pct_change().dropna()
        trading_return = returns.mean()
        trading_vol = returns.std()
        daily_interest = interest_rate / 252
        sharpe = (trading_return - daily_interest) / trading_vol
    else:
        benchmark_return = dataframe["Returns"].mean()
        benchmark_vol = dataframe["Returns"].std()
        daily_interest = interest_rate / 252
        sharpe = (benchmark_return - daily_interest) / benchmark_vol

    return sharpe














def coint_pairs(dataframe, sector_mean):
    """
    This function returns the pairs with minimum p-value of the cointegrated test
    :param datadict: dictionary object contains all the data
    :param p_mean: the average adjusted close prices for each ETF over a certain period
    :return: the
    """

    columns = dataframe.columns.values
    cointegration = []
    n = len(columns)
    # specify the minimum p_value
    min_pvalue = 1.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            X = dataframe.iloc[:, i].values
            Y = dataframe.iloc[:, j].values
            model = sm.OLS(Y, X).fit()
            predictions = model.predict(X)
            res = Y - predictions
            cadf = ts.adfuller(res)
            pvalue = cadf[1]

            # take the pairs with lowest p-values
            if (pvalue < min_pvalue):
                min_pvalue = pvalue
                if sector_mean[j] > sector_mean[i]:
                    # index with lower average price ticker in front
                    index = (columns[i], columns[j])
                else:
                    index = (columns[j], columns[i])

    cointegration.append(index)
    # Notice that the last element in the cointegration has the minimum p-value
    return cointegration


def momentum_sort(data):
    """
    This function receives the dataframe as the input.
    It returns the pair of indices with lowest and highest cumulative returns.
    The first symbol in the list stands for the asset with lowest returns
    """
    cum_return = (1 + data.pct_change().dropna()).cumprod().iloc[-1]
    pairs = [cum_return.idxmin(), cum_return.idxmax()]
    # return index
    return pairs


def s_signal(data, pairs):
    """
    perform the AR(1) model for the cointegration residual
    assume the residual is staionary
    cointegration
    """
    # obtain residuals and first run the linear regression to obtain residuals
    res = linear_resid(data, pairs)
    # perform the AR(1) to residuals (m is the mean of residual)
    m, std = moving_avg(res)
    #
    s = (res[-1] - m)/std
    return s


def mean_reversion(s, wealth, pl, ph, low_thresh_0, high_thresh_0):
    """
    s: strength of mean reversion signal
    wealth: dollar positions (portfolio values)
    pl : the price of the lower priced stock
    ph : the price of the higher priced stock
    low_thresh_0 : minimum threshold between the prices of these two cointegrated stocks (-1.25 std)
    high_thresh_0: maximum threshold between the prices of these two cointegrated stocks (1.25 std)
    """

    flag = False
    if (s < low_thresh_0):
        # s will probability increase in future
        p1 = - wealth / pl  # short
        p2 = 2. * wealth / ph  # long
        position = [p1, p2]

    elif (s > high_thresh_0):
        # s will probably decrease in future
        p1 = 2. * wealth / pl
        p2 = - wealth / ph
        position = [p1, p2]

    else:
        # no portfolio built
        position = [0., 0.]
        flag = True
    return position, flag


def momentum_trade(pl, ph, wealth):
    """
    Momentum trade (long high value stocks and short low value stock)
    """
    p1 = - wealth/pl
    p2 = 2.*wealth/ph
    position = [p1, p2]
    flag = False
    return position, flag


# close position when switch regime or mean reversion disappear
def close(position, pl, ph, end_t, wealth_path):
    """
    close the position when there are regime switches.
    """
    eps = 0.001
    wealth = position[end_t][0] * pl + position[end_t][1] * ph

    # if there are no stock positions, we copy the wealth from last time
    if (position[end_t][0] < eps and position[end_t][1] < eps):
        wealth = wealth_path[end_t]
    return wealth


def wealth_cal(position, pl, ph, time):
    """
    Calculate the dollar positions of our investments
    """
    return position[time][0]*pl + position[time][1]*ph


def ppair(t, pairs, prices):
    """
    We first retrieve the names of the stocks and obtain their prices
    """
    l = pairs[0][0]
    h = pairs[0][1]
    pl = prices.iloc[t][l]
    ph = prices.iloc[t][h]
    return pl, ph


def moving_avg(res):
    """
    obtain the moving average of mean and standard deviation
    using ARIMA model.
    """
    model = ARIMA(res, order = (1,0,0))
    model_fit = model.fit(disp=0)
    coef = model_fit.params
    residuals = pd.DataFrame(model_fit.resid)
    #print(model_fit.summary())
    sigma_e = residuals.values.std(ddof=1)
    m = coef[0]/(1-coef[1])
    sigma = np.sqrt(sigma_e**2./(1-coef[1]**2.))
    return m, sigma


def linear_resid(data, pairs):
    """
    compute the residuals using linear regressions
    """
    i = pairs[0][0]
    j = pairs[0][1]
    X = data.iloc[:, i].values
    Y = data.iloc[:, j].values
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)
    res = Y - predictions
    return res

