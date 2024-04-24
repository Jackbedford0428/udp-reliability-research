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
from tqdm.notebook import tqdm
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
# # Set Paramaters

# %%
# modeling
dirc_mets_lst = ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl']
anchor_mode_lst = ['by_event', 'by_packet']
test_mode = False

# %% [markdown]
# # Dual Radio Profiling

# %% [markdown]
# # Generate DR Model ID

# %%
sr_model_name = '20240417_1333007d66_new_data_sync_v2'
sr_model_id = sr_model_name[:19] if len(sr_model_name) > 19 else sr_model_name
sr_model_dscp = sr_model_name[20:] if len(sr_model_name) > 19 else None

dr_model_id = 'dr_20240417_1452001e84'
# dr_model_id = 'dr_' + model_identity()

print('SR Model ID:', sr_model_id, sr_model_dscp)
print('DR Model ID:', dr_model_id)
# print('DR Model ID:', dr_model_id, dr_model_dscp)

# %%
for anchor_mode in anchor_mode_lst:
    dr_model_dscp = f'anchor_{anchor_mode}'
    dr_model_name = f'{dr_model_id}_{dr_model_dscp}'
    print('DR Model ID:', dr_model_name)

# %% [markdown]
# # BR Model

# %%
# Dual Radio Example
# dates = data_loader(query_dates=True, show_info=True)

# selected_dates = [s for s in dates if s >= '2023-09-12']
selected_dates = ['2024-03-19']
# excluded_dates = []
# selected_exps = []
# excluded_exps = ['Modem_Action_Test', 'Control_Group', 'Control_Group2', 'Control_Group3']
selected_routes = ['BR']
# excluded_routes = []
filepaths = data_loader(mode='dr', selected_dates=selected_dates, selected_routes=selected_routes)

print(len(filepaths))
# pprint(filepaths)

# %%
for anchor_mode in anchor_mode_lst:
    dr_model_dscp = f'anchor_{anchor_mode}'
    dr_model_name = f'{dr_model_id}_{dr_model_dscp}'
    print('DR Model ID:', dr_model_name)
    for dirc_mets in dirc_mets_lst:
        dr_model = DrProfile(filepaths, 'BR',
                             sr_model_id, sr_model_dscp, dr_model_id, dr_model_dscp, dirc_mets=dirc_mets,
                             anchor_mode=anchor_mode, test_mode=test_mode)

# %% [markdown]
# # A Model

# %%
# Dual Radio Example
# dates = data_loader(query_dates=True)

# selected_dates = [s for s in dates if s >= '2023-09-12']
selected_dates = ['2024-03-20']
# excluded_dates = []
# selected_exps = []
# excluded_exps = ['Modem_Action_Test', 'Control_Group', 'Control_Group2', 'Control_Group3']
selected_routes = ['A']
# excluded_routes = []
filepaths = data_loader(mode='dr', selected_dates=selected_dates, selected_routes=selected_routes)

print(len(filepaths))
# pprint(filepaths)

# %%
for anchor_mode in anchor_mode_lst:
    dr_model_dscp = f'anchor_{anchor_mode}'
    dr_model_name = f'{dr_model_id}_{dr_model_dscp}'
    print('DR Model ID:', dr_model_name)
    for dirc_mets in dirc_mets_lst:
        dr_model = DrProfile(filepaths, 'A',
                             sr_model_id, sr_model_dscp, dr_model_id, dr_model_dscp, dirc_mets=dirc_mets,
                             anchor_mode=anchor_mode, test_mode=test_mode)

# %%



