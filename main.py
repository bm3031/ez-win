import numpy as np
import pandas as pd
from pandas_datareader import data
from pandas_datareader.stooq import StooqDailyReader as reader
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import cloudscraper
import requests_cache
from datetime import datetime, timedelta
import os
import pickle

blacklist = ['SPIR', 'EWCZ', 'SRZN','PBHC']
small_tickers = []
micro_tickers = []

scraper = cloudscraper.create_scraper()

small1 = scraper.get(
    "https://finviz.com/screener.ashx?v=111&f=cap_small,sh_price_o2,ta_sma20_pa10,ta_sma50_pa10&ft=4&o=-dividendyield").text

soup1 = BeautifulSoup(small1, "html.parser")
# print(soup)
small_table1 = soup1.find(text="No.").find_parent("table")
# print(small_table)
small_rows1 = small_table1.find_all('tr')

c = 0
for row in small_rows1:

    # print(row)
    cols = row.find_all('td')
    # print(cols)
    ticker = cols[1].text

    if ticker not in blacklist and ticker != 'Ticker':
        small_tickers.append(ticker)
        c += 1

small2 = scraper.get(
    "https://finviz.com/screener.ashx?v=111&f=cap_small,sh_price_o2,ta_sma20_pa10,ta_sma50_pa10&ft=4&o=-dividendyield&r=21").text

soup2 = BeautifulSoup(small2, "html.parser")
# print(soup2)
small_table2 = soup2.find(text="No.").find_parent("table")
# print(small_table)
small_rows2 = small_table2.find_all('tr')


for row in small_rows2:
    if c < 25:
        # print(row)
        cols = row.find_all('td')
        # print(cols)
        ticker = cols[1].text

        if ticker not in blacklist and ticker != 'Ticker':
            small_tickers.append(ticker)
            c += 1

# print(small_tickers)



micro1 = scraper.get(
    "https://finviz.com/screener.ashx?v=111&f=cap_micro,fa_pb_u1,sh_price_o2,ta_sma20_pa,ta_sma50_pa&o=-perf13w").text

soup3 = BeautifulSoup(micro1, "html.parser")
# print(soup)
micro_table1 = soup3.find(text="No.").find_parent("table")
# print(small_table)
micro_rows1 = micro_table1.find_all('tr')

c = 0
for row in micro_rows1:

    # print(row)
    cols = row.find_all('td')
    # print(cols)
    ticker = cols[1].text

    if ticker not in blacklist and ticker != 'Ticker':
        micro_tickers.append(ticker)
        c += 1

micro2 = scraper.get(
    "https://finviz.com/screener.ashx?v=111&f=cap_micro,fa_pb_u1,sh_price_o2,ta_sma20_pa,ta_sma50_pa&o=-perf13w&r=21").text

soup4 = BeautifulSoup(micro2, "html.parser")
# print(soup2)
micro_table2 = soup4.find(text="No.").find_parent("table")
# print(small_table)
micro_rows2 = micro_table2.find_all('tr')

for row in micro_rows2:
    if c < 25:
        # print(row)
        cols = row.find_all('td')
        # print(cols)
        ticker = cols[1].text

        if ticker not in blacklist and ticker != 'Ticker':
            micro_tickers.append(ticker)
            c += 1


print(small_tickers)
print(micro_tickers)

tickers = (small_tickers[0:15] + micro_tickers[0:15])

print(tickers)

# OPTIMIZATION

#df = data.DataReader(tickers, 'yahoo', start='2021/08/19', end='2021/09/18')["Adj Close"]

end = datetime.now()

start = end - timedelta(days=90)

end = end.strftime('%Y/%m/%d')
start = start.strftime('%Y/%m/%d')

if os.path.exists("cache.dat"):
    with open("cache.dat","rb") as f:
        df = pickle.load(f)

else:
    df = reader(tickers, start=start, end=end).read()['Close']
    with open("cache.dat","wb") as f:
        pickle.dump(df,f)
    print("Loaded data from cache")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)

# logs of percentage change for time additive returns and covariance matrix

cov_matrix = df.pct_change().apply(lambda x: np.log(1 + x)).cov()

# individual expected returns based off of last month's returns

ind_er = 100*(df.iloc[0]/df.iloc[-1]-1)

# print(ind_er)
# print(len(df.columns))

p_ret = []  # Define an empty list for portfolio returns
p_vol = []  # Define an empty list for portfolio volatility
p_weights = []  # Define an empty list for asset weights

num_assets = len(df.columns)
num_portfolios = 100000  # number of points on graph

for portfolio in range(num_portfolios):

    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)

    p_weights.append(weights)
    # print(weights.sum())
    returns = np.dot(weights, ind_er)  # expected returns

    p_ret.append(returns)

    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()  # Portfolio Variance
    sd = np.sqrt(var)  # Daily standard deviation
    ann_sd = sd * np.sqrt(365)  # Monthly standard deviation = volatility
    p_vol.append(ann_sd)

# print(p_ret, "/n", p_vol)
data1 = {'Returns': p_ret, 'Volatility': p_vol}
# print(data)

for counter, symbol in enumerate(df.columns.tolist()):
    # print(counter, symbol)
    data1[symbol+' weight'] = [w[counter] for w in p_weights]

#  print(data)

portfolios = pd.DataFrame(data1)  # Dataframe of the portfolios created
# print(portfolios)
# Plot efficient frontier
portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10, 10])

# Finding the optimal portfolio
rf = 15  # risk factor - target

optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]  # Cum ratio

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(optimal_risky_port)

for i in range(2, 32):
    #  print(df.iloc[-1, i-2])
    optimal_risky_port[i] = (99700 * optimal_risky_port[i])/df.iloc[0, i-2]


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(optimal_risky_port)

plt.subplots(figsize=(10, 10))
plt.scatter(portfolios['Volatility'], portfolios['Returns'], marker='o', s=10, alpha=0.3)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)

print("Cum ratio:", ((portfolios['Returns']-rf)/portfolios['Volatility']).max())

# market_returns = data.DataReader('SPY', 'yahoo', start='2021/08/19', end='2021/09/18').head()["Adj Close"].head()
# market_er = 100*(market_returns.iloc[-1]/market_returns.iloc[0]-1)
# daily_change = market_returns.pct_change().apply(lambda x: np.log(1 + x))
# print(daily_change.std())
# print(market_er)
# print("S&P 500 Cum ratio:", (market_er-rf)/daily_change.std())


plt.show()
