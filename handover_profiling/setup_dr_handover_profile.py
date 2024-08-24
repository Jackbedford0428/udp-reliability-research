# %% [markdown]
# # Import Modules & Util Functions

# %%
import os
import sys
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
# import seaborn as sns

from scipy.stats import gaussian_kde
from tqdm import tqdm
from pprint import pprint
from pytictoc import TicToc
import argparse

from Toolkit import *
from HandoverProfile import *

# Configure display options
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 100)
# %config InlineBackend.figure_format = 'retina'

# Set plot style
# plt.style.use('ggplot')

# %% [markdown]
# # Set Paramaters

# %%
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--route", type=str, help="experiment route")
parser.add_argument("-n1", "--model_name1", type=str, help="model name layer 1")
parser.add_argument("-n2", "--model_name2", type=str, help="model name layer 2")
parser.add_argument("-d", "--dates", type=str, nargs='+', help="date folders to process")
parser.add_argument("-m", "--metrics", type=str, help="direction and metrics")
parser.add_argument("-am", "--anchor_mode", type=str, help="anchor mode")
parser.add_argument("-t", "--test_mode", action="store_true", help="test mode or not")
args = parser.parse_args()


if args.model_name1 is None:
    raise ValueError('Missing Value: model_name1')

model_name1 = args.model_name1
print('Model Name L1:', model_name1)

# { all }
# { BR, A, B, R, G, O1, O2, Y }
# { subBR, subA, subB, subR, subG, subO1, subO2, subY }
route = 'BR' if args.route is None else args.route
model_name2 = route if args.model_name2 is None else args.model_name2
print('Model Name L2:', model_name2)

if args.route is None:
    selected_routes = ['BR']
else:
    if args.route == 'all':
        selected_routes = ['BR', 'A', 'B', 'R', 'G', 'O1', 'O2', 'Y']
    elif 'sub' in args.route:
        # setup
        selected_routes = ['BR', 'A', 'B', 'R', 'G', 'O1', 'O2', 'Y']
        selected_routes.remove(args.route[3:])
    else:
        selected_routes = [args.route]
print('Setup/Eval Routes:', route)
print('Selected Routes:', selected_routes)

selected_dates = data_loader(query_dates=True) if args.dates is None else args.dates
print('Selected Dates:', selected_dates)

# { dl_lost, dl_excl, ul_lost, ul_excl }
dirc_mets = 'dl_lost' if args.metrics is None else args.metrics
print('Performance Metrics:', dirc_mets)

# { by_event, by_packet, by_mixed }
anchor_mode = 'by_event' if args.anchor_mode is None else args.anchor_mode
print('Anchor Mode:', anchor_mode)

# take 1 data only
test_mode = args.test_mode
print('Test Mode:', test_mode)


# %% [markdown]
# # Dual Radio Profiling

# %%
print('----------------')
filepaths = data_loader(mode='dr', selected_dates=selected_dates, selected_routes=selected_routes)
print('Number of Files:', len(filepaths))
# pprint(filepaths)
print('----------------')

dr_model = DrProfile(filepaths, model_name1, model_name2, route,
                     dirc_mets, anchor_mode, test_mode)