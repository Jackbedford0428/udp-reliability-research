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
from tqdm.notebook import tqdm
from pprint import pprint
from pytictoc import TicToc
import argparse

from myutils import *
from profile import *

# Configure display options
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 100)
# %config InlineBackend.figure_format = 'retina'

# Set plot style
# plt.style.use('ggplot')

# %% [markdown]
# # Add arguments

# %%
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dates", type=str, nargs='+', help="date folders to process")
parser.add_argument("-r", "--route", type=str, help="experiment route")
parser.add_argument("-s", "--slice", type=int, help="slice number for testing functionality")
parser.add_argument("-srm", "--sr_model", type=str, help="SR model_name")
parser.add_argument("-drm", "--dr_model", type=str, help="DR model_name")
parser.add_argument("-dm", "--direction_metrics", type=str, default='dl_lost', help="direction and metrics")
parser.add_argument("-am", "--anchor_mode", type=str, default='by_event', help="anchor mode")
parser.add_argument("-cc", "--corr_coef", type=str, default=None, help="correlation coefficient")
parser.add_argument("-it", "--iteration", type=int, default=5, help="iteration number")
parser.add_argument("-dt", "--dataset_type", type=str, default="train", help="dataset type")
parser.add_argument("-sa", "--save_answer", action="store_true", help="save answer or not")
args = parser.parse_args()

# %% [markdown]
# # Set Paramaters

# %%
# evaluate
dirc_mets = args.direction_metrics
anchor_mode = args.anchor_mode
args.corr_coef = None if args.corr_coef == 'None' else args.corr_coef
model_corr = args.corr_coef if args.corr_coef is None or args.corr_coef == 'adjust' else f'{args.corr_coef}_cc'
iter_num = args.iteration
save_answer = args.save_answer

print(dirc_mets, anchor_mode, model_corr, iter_num, save_answer)

# %% [markdown]
# # Dual Radio Evaluation

# %% [markdown]
# # Enter Model IDs

# %%
sr_model_name = args.sr_model
dr_model_id = args.dr_model

sr_model_id = sr_model_name[:19] if len(sr_model_name) > 19 else sr_model_name
sr_model_dscp = sr_model_name[20:] if len(sr_model_name) > 19 else None
dr_model_dscp = f'anchor_{anchor_mode}'
dr_model_name = f'{dr_model_id}_{dr_model_dscp}'

print('SR Model:', sr_model_id, sr_model_dscp)
print('DR Model:', dr_model_id, dr_model_dscp)

# %% [markdown]
# # BR: DR Eval

# %%
# Dual Radio Example
# dates = data_loader(query_dates=True, show_info=True)

# selected_dates = [s for s in dates if s >= '2023-09-12']
selected_dates = args.dates
# excluded_dates = []
# selected_exps = []
# excluded_exps = ['Modem_Action_Test', 'Control_Group', 'Control_Group2', 'Control_Group3']
selected_routes = [args.route]
# excluded_routes = []
filepaths = data_loader(mode='dr', selected_dates=selected_dates, selected_routes=selected_routes)

if args.slice is not None:
    filepaths = filepaths[:args.slice]
print(len(filepaths))
# pprint(filepaths)

# # %%
print('=======================', model_corr, '=======================')
eval = DrEval(filepaths, args.route, model_corr, sr_model_id, sr_model_dscp, dr_model_id, dr_model_dscp, dirc_mets=dirc_mets,
            anchor_mode=anchor_mode, save_answer=save_answer, dataset_type=args.dataset_type)
eval.run_hist_method(N=iter_num)
eval.plot()