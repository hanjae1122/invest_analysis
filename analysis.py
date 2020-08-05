import os
import datetime
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
# from openpyxl import load_workbook

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100)

PLOT_PATH = 'plots'
HIST_PATH = 'data/historic'
ECON_PATH = 'data/economic'
PLOT_PARAMS = {
    'palette': ['#58D68D', '#E74C3C', '#D35400', '#34495E']
}

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


############################################################
# Read data
############################################################
# df_econ = pd.read_csv(os.path.join(ECON_PATH, 'agg_nonindexed.csv'),
#                       parse_dates=['DATE'])
# df_econ['DATE'] = df_econ['DATE'].dt.to_period('m')
# df_econ.set_index('DATE', inplace=True)

df_prices = pd.read_csv(os.path.join(HIST_PATH, 'ticks_M.csv'),
                        parse_dates=['DATE'])
df_prices['DATE'] = df_prices['DATE'].dt.to_period('m')
df_prices.set_index('DATE', inplace=True)

############################################################
# Backtest returns
############################################################
# analyze returns
allocs_for_analysis = [
    {'VTI': 15, 'VEA': 6, 'VWO': 4,
     'TLT': 25, 'ITE': 20,
     'TIP': 15,
     'DBC': 10, 'GLD': 5},
    # {'VTI': 9, 'VTV': 2, 'VOE': 2, 'VBR': 2,
    #  'VEA': 6, 'VWO': 4,
    #  'TLT': 25, 'ITE': 20,
    #  'TIP': 15,
    #  'DBC': 10, 'GLD': 5},
    {'VTI': 60,
     'TLT': 40},
    {'VTI': 10.22820274,
     'VEA': 2.80703365, 'VWO': 1.45880438,
     'TLT': 12.1659097, 'ITE': 36.72334412,
     'TIP': 24.69659242,
     'DBC': 7.40656824, 'GLD': 4.51354474},
    {'VTI': 12, 'VEA': 3.6, 'VWO': 2.4,
     'TLT': 18, 'ITE': 30,
     'TIP': 18,
     'DBC': 9.6, 'GLD':6}
]

# determine what ticks we need
unique_ticks = list(set([k for d in allocs_for_analysis for k in d.keys()]))
df_allprcs = df_prices[unique_ticks].dropna(axis=0, how='any')

# narrow by dates
df_allprcs = df_allprcs[(df_allprcs.index > '2000-01-01') & (df_allprcs.index < '2012-01-01')]
num_months = df_allprcs.shape[0]

# get cum rtn for each asset
df_allcum_rtns = df_allprcs / df_allprcs.iloc[0]

# plot cum rtn
fig, ax = plt.subplots(figsize=(7,7))
df_allcum_rtns.reset_index().plot(x='DATE', lw=0.6,  ax=ax)

# plot each strategy
palette = iter(plt.cm.rainbow(np.linspace(0, 1, len(allocs_for_analysis))))
for alloc_num, alloc in enumerate(allocs_for_analysis):
    # get ticks and % allocations for each tick
    ticks, pct_allocs = [], []
    for (k, v) in alloc.items():
        ticks.append(k)
        pct_allocs.append(v)
    # normalize % allocation
    pct_allocs = np.array(pct_allocs)/sum(pct_allocs)
    
    print('==================================================')
    print(alloc)
    df_prcs = df_allprcs[ticks]

    portfolio_value = [10**5]
    for i in range(num_months - 1):
        # calculate # shares per tick
        shares = (portfolio_value[-1] * pct_allocs / df_prcs.iloc[i]).values
        # calculate total portfolio value
        portfolio_value.append(sum(df_prcs.iloc[i + 1].values * shares))
    portfolio_value = np.array(portfolio_value)
    
    # calculate returns
    cum_rtns = pd.Series(portfolio_value / portfolio_value[0],
                     index=df_prcs.index)
    rtns = (portfolio_value[1:] / portfolio_value[:-1] - 1)
    label = ('{3} T: {4:.3f} M:{1:.3f} \nS: {1:.3f}, M/S: {2:.3f}'
             .format(rtns.mean(), rtns.std(),
                     rtns.mean()/rtns.std(),
                     alloc_num, cum_rtns[-1]))
    # plot returns
    cum_rtns.plot(label=label,
                  c=next(palette),
                  # cmap=plt.cm.get_cmap('tab10'),
                  ax=ax)
    
ax.legend(loc='best', ncol=2)    
fig.tight_layout()
fig.savefig(os.path.join(PLOT_PATH, 'june performance.png'.format(alloc_num)))         
plt.close('all')

############################################################
# Annualized volatility
############################################################
df_prices = pd.read_csv(os.path.join(HIST_PATH, 'ticks_M.csv'),
                        parse_dates=['DATE'])
df_prices['DATE'] = df_prices['DATE'].dt.to_period('m')
df_prices.set_index('DATE', inplace=True)
