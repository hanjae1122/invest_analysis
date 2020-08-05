import os
import importlib
import math
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100)

os.chdir('C:\\Users\\Jae\\source\\repos\\Invest')
#%%

#################### temporary stuff ####################
def get_lagged_df(df, i=1):
    if i == 0:
        return df
    return pd.DataFrame(df[:-i].values, index=df.index[i:], columns=df.columns)


def anchored_df(df, first_date=None, normalize=False):
    # remove all leading nans
    first_valid_indices = []
    for column in df:
        first_valid_indices.append(df[column].first_valid_index())
    earliest_date = max(first_valid_indices)
    if first_date:
        first_date = pd.Period(first_date, 'm')
        if first_date < earliest_date:
            print('First date {0} too early. Using earliest available date {1}'
                  .format(first_date, earliest_date))
            reduced_df = df[earliest_date:]
        reduced_df = df[first_date:]
    else:
        reduced_df = df[earliest_date:]

    print('Reduced rows from {0} to {1}'.format(df.shape[0],
                                                reduced_df.shape[0]))
    if normalize:
        # return normalized values
        return (reduced_df - reduced_df.values[0])/reduced_df.std().values
    else:
        return reduced_df - reduced_df.values[0]


df = df_prices.merge(df_econ, left_index=True, right_index=True)

temp = anchored_df(df[['PCY', 'rGDP', 'UNEMP', 'CPI']], normalize=True)
temp.plot(figsize=(10, 10))
plt.show()

df_prc_chng  = pd.DataFrame(df_stocks[1:].values/df_stocks[:-1].values-1,
                            columns=df_stocks.columns,
                            index=df_stocks.index[1:])

############################################################
# Correlation analysis
############################################################
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# correlation btw assets
# define corr variables
# Generate a mask for the upper triangle
D = df_prc_chng.shape[1]
mask = np.zeros((D, D), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
plt.figure(figsize=(10, 10))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(df_prc_chng.corr(), mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.legend()
plt.tight_layout()
plt.show()

# reduced correlations
D = df_prc_chng.shape[1]
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Set up the matplotlib figure
f = plt.figure(figsize=(10, 10))
for i in range(4):
    f.add_subplot(int('22'+str(i+1)))
    lag_df_econ = get_lagged_df(df_econ, i)
    df = pd.concat([df_prc_chng, lag_df_econ], axis=1, join='inner')
    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(df.corr().iloc[:D, D:], cmap=cmap, vmax=.3, center=0,
                     square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_title('Lag {0}'.format(i))
plt.legend()
plt.tight_layout()
plt.show()

#%%
############################################################
# Check current holdings
############################################################
allocs = [
    {'SCHB': 15, 'GXC': 2, 'EWJ': 3, 'VGK': 3, 'IEMG': 2,
     'TLT': 25, 'SCHR': 20, 'SCHP': 15, 'DBC': 5,
     'PDBC': 5, 'BAR': 5},
    {'SCHB': 15, 'VEA': 6, 'IEMG': 4,
     'TLT': 25, 'SCHR': 20,
     'SCHP': 15,
     'DBC': 5, 'PDBC': 5, 'BAR': 5},
    # allocation by risk parity
    # 25 stocks, 25 bonds, 25 tips, 25 comm
    # stable allocation in post-2007 period
    {'SCHB': 10, 'VEA': 3, 'IEMG': 2,
     'TLT': 12, 'SCHR': 30,
     'SCHP': 30,
     'DBC': 4, 'PDBC': 4, 'BAR': 5},
    {'SCHB': 10, 'VEA': 3, 'IEMG': 2,
     'TLT': 12, 'SCHR': 30,
     'SCHP': 20,
     'DBC': 4, 'PDBC': 4, 'BAR': 5},
    {'SCHB': 10, 'VEA': 3, 'IEMG': 2, 'GXC': 5,
     'TLT': 12, 'SCHR': 30,
     'SCHP': 25,
     'DBC': 4, 'PDBC': 4, 'BAR': 5}
]

l = []
with open('holdings.txt') as f:
    for s in f.readlines():
        if s == '—\n':
            l.append(None)
        else:
            ss = s.strip().replace('$', '').replace(',', '')
            try:
                l.append(float(ss))
            except ValueError:
                l.append(ss)
colnames, data = l[:7], l[7:]
columns = []
for start_index in range(7):
    columns.append(data[start_index::7])
dict4df = {}
for i, colname in enumerate(colnames):
    dict4df[colname] = columns[i]
df_holdings = pd.DataFrame(dict4df).set_index('Symbol')
df_holdings['Current % alloc'] = (100*df_holdings['Equity']/
                                  df_holdings['Equity'].sum())
#%%
indv_stocks = []
# stocks
indv_stocks_bal = sum(df_holdings.loc[indv_stocks, 'Equity'].astype('float64'))
remaining_bal = sum(df_holdings['Equity']) - indv_stocks_bal
remaining_bal = 0.85*65387
ticks, pct_allocs = [], []
for (k, v) in allocs[4].items():
    ticks.append(k)
    pct_allocs.append(v)
pct_allocs = np.round(np.array(pct_allocs)/sum(pct_allocs) * 100, 2)
print('==================================================')
print(pd.DataFrame({'Symbol': ticks, '%': pct_allocs}))
# for tick in alloc[0]:
# prc_data = get_data(alloc[0], 'D')
close = df_holdings.loc[ticks, 'Price'].tolist()
shares = [int(round(e)) for e in list(remaining_bal * np.array(pct_allocs) /
                                      (100 * np.array(close)))]
port = np.array(close) * np.array(shares)
print('Remaining balance: {0:.2f}\n% allocs: {2}\nPrices: {3}\nPorVal: {5}\nStocks: {1}\nShares: {4}'.format(remaining_bal, ticks, pct_allocs, close, shares, port))

allocation = pd.DataFrame({'% allocs': pct_allocs, 'Prices': close,
                           'Stocks': ticks, 'PorVal': port,
                           'Actual % alloc': np.round(port/port.sum() * 100, 2),
                           'Shares': shares})
print(allocation['PorVal'].sum())
print(allocation)
