# %% [markdown]
# # Import Modules & Util Functions

# %%
import os
import sys
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns

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
parser.add_argument("-d", "--dates", type=str, nargs='+', help="date folders to process")
parser.add_argument("-r", "--route", type=str, help="experiment route")
parser.add_argument("-s", "--slice", type=int, help="slice number for testing functionality")
parser.add_argument("-p", "--model_path", type=str, help="model_path")
parser.add_argument("-m", "--model_name", type=str, help="model_name")
parser.add_argument("-mm", "--direction_metrics", type=str, default='dl_lost', help="direction and metrics")
parser.add_argument("-am", "--anchor_mode", type=str, default='by_event', help="anchor mode")
parser.add_argument("-cc", "--corr_coef", type=str, default='mle', help="correlation coefficient")
parser.add_argument("-it", "--iteration", type=int, default=1, help="iteration number")
parser.add_argument("-dt", "--dataset_type", type=str, default="train", help="dataset type")
parser.add_argument("-tt", "--test_mode", action="store_true", help="test_mode")
args = parser.parse_args()

dirc_mets = args.direction_metrics
anchor_mode = args.anchor_mode
model_corr = args.corr_coef if args.corr_coef == 'mle' or args.corr_coef == 'adjust' else f'{args.corr_coef}_cc'
iter_num = args.iteration
# save_answer = args.save_answer
dataset_type = args.dataset_type
model_name = args.model_name
model_path = args.model_path
test_mode = args.test_mode

print(dirc_mets, anchor_mode, model_corr, iter_num)

# %% [markdown]
# # BR: DR Eval

# %%
# Dual Radio Example
if args.dates is not None:
    selected_dates = args.dates
else:
    selected_dates = data_loader(query_dates=True)
    # selected_dates = []

if args.route is not None:
    if args.route == 'all':
        selected_routes = ['BR', 'A', 'B', 'R', 'G', 'O2']
    elif 'sub' in args.route:
        rm_element = args.route[3:]
        selected_routes = ['BR', 'A', 'B', 'R', 'G', 'O2']
        selected_routes.remove(rm_element)
    else:
        selected_routes = [args.route]
else:
    route = model_name.replace('sub', '')
    selected_routes = [route]
    # selected_routes = ['BR']
# route = args.route if args.route is not None else 'BR'
route = args.route

filepaths = data_loader(mode='dr', selected_dates=selected_dates, selected_routes=selected_routes)

if args.slice is not None:
    filepaths = filepaths[:args.slice]

print(selected_routes)
print(len(filepaths))
# pprint(filepaths)

# # %%
print('=======================', model_corr, '=======================')
# dirc_mets_list = ['dl_lost', 'ul_lost', 'dl_excl', 'ul_excl']
# for dirc_mets in dirc_mets_list:
eval = DrEval(filepaths, route, dataset_type,
        model_path, model_name, anchor_mode,
        dirc_mets=dirc_mets, model_corr=model_corr,
        sp_columns=['type'], ts_column='Timestamp', w_size=0.01,
        iter_num=iter_num, test_mode=test_mode)
# eval = DrEval(filepaths, args.route, model_corr, sr_model_id, sr_model_dscp, dr_model_id, dr_model_dscp, dirc_mets=dirc_mets,
#             anchor_mode=anchor_mode, save_answer=save_answer, dataset_type=args.dataset_type)
eval.run_hist_method(N=iter_num)
eval.plot()


# python3 eval_dr_handover_profile.py -r A -m all -dt test -mm dl_lost -cc mle -p 20240630_latest_model -am by_event -it 1 -d 2024-03-19 2024-03-20 2024-06-18-1 2024-06-19-1 2024-06-20-1 2024-06-21-1 -tt