import os
import datetime
import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import optimize
# from openpyxl import load_workbook

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 100)

PLOT_PATH = 'plots'
HIST_PATH = 'data/WRDS'
ECON_PATH = 'data/economic'

def plot_path(filename):
    return os.path.join(PLOT_PATH, filename)

############################################################
# Read data
############################################################
df_econ = pd.read_csv(os.path.join(ECON_PATH, 'agg_nonindexed.csv'),
                      parse_dates=['DATE'])
df_econ['DATE'] = df_econ['DATE'].dt.to_period('m')
df_econ.set_index('DATE', inplace=True)

df_prices = pd.read_csv(os.path.join(HIST_PATH, 'rets_D.csv'),
                        parse_dates=['DATE'])
df_prices['DATE'] = df_prices['DATE'].dt.to_period('d')
df_prices.set_index('DATE', inplace=True)

############################################################
# Risk contribution
############################################################
def pct_risk_contribution(allocs, cov_rtns):
    portfolio_vol = np.sqrt(np.dot(np.dot(allocs, cov_rtns), allocs))
    return np.multiply(allocs, np.dot(cov_rtns, allocs)) / (portfolio_vol ** 2)
    
def risk_contribution_diff(allocs, cov_rtns, target_risk):    
    pct = pct_risk_contribution(allocs, cov_rtns)
    return ((pct - target_risk) ** 2).sum()


# ETFs for analysis
ticks = ['VTI', 'VEA', 'VWO', 'TLT', 'ITE', 'TIP', 'DBC', 'GLD']
init_allocs = [15, 6, 4, 25, 20, 15, 10, 5]
target_risk = [15, 6, 4, 15, 10, 25, 15, 10]

# normalize % allocation
init_allocs = np.array(init_allocs)/sum(init_allocs)
target_risk = np.array(target_risk)/sum(target_risk)

# calculate returns and covariance matrix
df_rtns = df_prices[ticks].dropna(axis=0, how='any')
df_rtns = df_rtns[df_rtns.index > pd.Period('2009-10-01', 'D')]

cov_rtns = df_rtns.cov()
# pd.DataFrame({'Tick': ticks,
#               '% Allocation': 100 * init_allocs,
#               '% Risk contribution': 100 * pct_risk_contribution(init_allocs, cov_rtns)})
optimal_allocs = optimize.minimize(risk_contribution_diff,
                                         x0=init_allocs,
                                         args=(cov_rtns, target_risk),
                                         method='SLSQP',
                                         constraints={'type': 'eq',
                                                      'fun': lambda x: x.sum() - 1})
print(pd.DataFrame({'Tick': ticks,
                    '% Allocation': 100 * optimal_allocs.x,
                    '% Risk contribution': 100 * pct_risk_contribution(optimal_allocs.x,
                                                                       cov_rtns)}))
# optimals.append(optimal_allocs.x)

############################################################
# TIP analysis
############################################################
tip = df_prices['TIP'].dropna(axis=0)
tip = (tip + 1).cumprod()
tip = (tip - tip.mean())/tip.std()
tip.plot(label='TIPS')

cpi = df_econ['CPI'].dropna(axis=0)
cpi = cpi[cpi.index >= tip.index[0].asfreq('M')]
cpi = (cpi - cpi.mean())/cpi.std()
cpi.plot(label='CPI')

plt.legend()
plt.savefig(plot_path('tips.png'))
plt.clf()
