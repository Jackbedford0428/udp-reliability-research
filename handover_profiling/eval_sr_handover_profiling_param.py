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
parser.add_argument("-m", "--model", type=str, help="model_name")
parser.add_argument("-dm", "--direction_metrics", type=str, default='dl_lost', help="direction and metrics")
parser.add_argument("-it", "--iteration", type=int, default=5, help="iteration number")
parser.add_argument("-dt", "--dataset_type", type=str, default="train", help="dataset type")
parser.add_argument("-sa", "--save_answer", action="store_true", help="save answer or not")
args = parser.parse_args()

# print(args.dates)
# print(args.route)
# print(args.slice)
# print(args.model)
# print(args.direction_metrics)
# print(args.iteration)
# print(args.save_answer)

# %% [markdown]
# # Set Parameters

# %%
dirc_mets = args.direction_metrics
iter_num = args.iteration
save_answer = args.save_answer

print(dirc_mets, iter_num, save_answer)

# %% [markdown]
# # Single Radio Evaluation

# %% [markdown]
# # Enter Model ID

# %%
model_name = args.model
model_id = model_name[:19] if len(model_name) > 19 else model_name
model_dscp = model_name[20:] if len(model_name) > 19 else None
print('Model ID:', model_id, model_dscp)

# %% [markdown]
# # BR Eval

# %%
# Single Radio Example
# dates = data_loader(query_dates=True, show_info=True)

# selected_dates = [s for s in dates if s >= '2023-09-12']
selected_dates = args.dates
# excluded_dates = []
# selected_exps = []
# excluded_exps = ['Modem_Action_Test', 'Control_Group', 'Control_Group2', 'Control_Group3']
selected_routes = [args.route]
# excluded_routes = []
filepaths = data_loader(mode='sr', selected_dates=selected_dates, selected_routes=selected_routes)

if args.slice is not None:
    filepaths = filepaths[:args.slice]
print(len(filepaths))
# pprint(filepaths)

# %%
eval = Eval(filepaths, args.route, model_id, model_dscp, dirc_mets=dirc_mets, save_answer=save_answer, dataset_type=args.dataset_type)
eval.run_hist_method(N=iter_num)
eval.plot()
