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

from myutils import *
from profile import *

# Configure display options
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 100)
# %config InlineBackend.figure_format = 'retina'

# Set plot style
# plt.style.use('ggplot')

# %% [markdown]
# # Set Parameters

# %%
# modeling
dirc_mets_list = ['dl_lost', 'ul_lost', 'dl_excl', 'ul_excl']
# dirc_mets = 'dl_lost'
epochs = 3
test_mode = False
# evaluate
# iter_num = 5
# save_answer = True

# %% [markdown]
# # Generate Model ID

# %%
model_id = model_identity()
model_dscp = 'new_data_sync_v4'
print('Model ID:', model_id, model_dscp)

# model_name = '20240417_1333007d66_new_data_sync_v2'
# model_id = model_name[:19] if len(model_name) > 19 else model_name
# model_dscp = model_name[20:] if len(model_name) > 19 else None
# print('Model ID:', model_id, model_dscp)

# %% [markdown]
# # Single Radio Profiling

# %% [markdown]
# # BR Models

# %%
# Single Radio Example
# dates = data_loader(query_dates=True, show_info=True)

# selected_dates = [s for s in dates if s >= '2023-09-12']
selected_dates = ['2024-03-19']
# excluded_dates = []
# selected_exps = []
# excluded_exps = ['Modem_Action_Test', 'Control_Group', 'Control_Group2', 'Control_Group3']
selected_routes = ['BR']
# excluded_routes = []
filepaths = data_loader(mode='sr', selected_dates=selected_dates, selected_routes=selected_routes)

print(len(filepaths))
# pprint(filepaths)

# %%
for dirc_mets in dirc_mets_list:
    model = Profile(filepaths, 'BR', model_id, model_dscp, dirc_mets=dirc_mets, epochs=epochs, test_mode=test_mode)

# %% [markdown]
# # A Models

# %%
# Single Radio Example
# dates = data_loader(query_dates=True)

# selected_dates = [s for s in dates if s >= '2023-09-12']
selected_dates = ['2024-03-20']
# excluded_dates = []
# selected_exps = []
# excluded_exps = ['Modem_Action_Test', 'Control_Group', 'Control_Group2', 'Control_Group3']
selected_routes = ['A']
# excluded_routes = []
filepaths = data_loader(mode='sr', selected_dates=selected_dates, selected_routes=selected_routes)

print(len(filepaths))
# pprint(filepaths)

# %%
for dirc_mets in dirc_mets_list:
    model = Profile(filepaths, 'A', model_id, model_dscp, dirc_mets=dirc_mets, epochs=epochs, test_mode=test_mode)

# # %% [markdown]
# # # Single Radio Evaluation

# # %% [markdown]
# # # Enter Model ID

# # %%
# # model_name = '20240415_1604003926_test'
# # model_id = model_name[:19] if len(model_name) > 19 else model_name
# # model_dscp = model_name[20:] if len(model_name) > 19 else None

# print('Model ID:', model_id, model_dscp)

# # %% [markdown]
# # # BR Eval

# # %%
# # Single Radio Example
# # dates = data_loader(query_dates=True, show_info=True)

# # selected_dates = [s for s in dates if s >= '2023-09-12']
# selected_dates = ['2024-03-19']
# # excluded_dates = []
# # selected_exps = []
# # excluded_exps = ['Modem_Action_Test', 'Control_Group', 'Control_Group2', 'Control_Group3']
# selected_routes = ['BR']
# # excluded_routes = []
# filepaths = data_loader(mode='sr', selected_dates=selected_dates, selected_routes=selected_routes)

# print(len(filepaths))
# # pprint(filepaths)

# # %%
# eval = Eval(filepaths, 'BR', model_id, model_dscp, dirc_mets=dirc_mets, save_answer=save_answer, test_mode=test_mode)
# eval.run_hist_method(N=iter_num)
# eval.plot()

# # %%
# eval = Eval(filepaths, 'BR', model_id, model_dscp, dirc_mets='dl_lost', save_answer=True)
# eval.run_hist_method(N=1)
# eval.plot()

# # %% [markdown]
# # # A Eval

# # %%
# # Single Radio Example
# # dates = data_loader(query_dates=True)

# # selected_dates = [s for s in dates if s >= '2023-09-12']
# selected_dates = ['2024-03-20']
# # excluded_dates = []
# # selected_exps = []
# # excluded_exps = ['Modem_Action_Test', 'Control_Group', 'Control_Group2', 'Control_Group3']
# selected_routes = ['A']
# # excluded_routes = []
# filepaths = data_loader(mode='sr', selected_dates=selected_dates, selected_routes=selected_routes)

# print(len(filepaths))
# # pprint(filepaths)

# # %%
# eval = Eval(filepaths, 'A', model_id, model_dscp, dirc_mets=dirc_mets, save_answer=save_answer, test_mode=test_mode)
# eval.run_hist_method(N=iter_num)
# eval.plot()
