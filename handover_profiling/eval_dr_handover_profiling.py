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
dirc_mets = 'ul_excl'
anchor_mode = 'by_packet'
test_mode = False
# evaluate
iter_num = 2
save_answer = True
corr_lst = [None, 'adjust']
corr_lst = ['max', 'zero', '25%', '50%', '75%']
print(corr_lst)

# %% [markdown]
# # Dual Radio Evaluation

# %% [markdown]
# # Enter Model IDs

# %%
sr_model_name = '20240417_1333007d66_new_data_sync_v2'
dr_model_id = 'dr_20240417_1452001e84'

sr_model_id = sr_model_name[:19] if len(sr_model_name) > 19 else sr_model_name
sr_model_dscp = sr_model_name[20:] if len(sr_model_name) > 19 else None
dr_model_dscp = f'anchor_{anchor_mode}'
dr_model_name = f'{dr_model_id}_{dr_model_dscp}'

print('SR Model:', sr_model_id, sr_model_dscp)
print('DR Model:', dr_model_id, dr_model_dscp)

# # %% [markdown]
# # # BR: DR Eval

# # %%
# # Dual Radio Example
# # dates = data_loader(query_dates=True, show_info=True)

# # selected_dates = [s for s in dates if s >= '2023-09-12']
# selected_dates = ['2024-03-19']
# # excluded_dates = []
# # selected_exps = []
# # excluded_exps = ['Modem_Action_Test', 'Control_Group', 'Control_Group2', 'Control_Group3']
# selected_routes = ['BR']
# # excluded_routes = []
# filepaths = data_loader(mode='dr', selected_dates=selected_dates, selected_routes=selected_routes)

# print(len(filepaths))
# # pprint(filepaths)

# # %%
# for mode in corr_lst:
#     model_corr = mode if mode == None or mode == 'adjust' else f'{mode}_corr'
#     print('=======================', model_corr, '=======================')
#     eval = DrEval(filepaths, 'BR', model_corr, sr_model_id, sr_model_dscp, dr_model_id, dr_model_dscp, dirc_mets=dirc_mets,
#               anchor_mode=anchor_mode, test_mode=test_mode, save_answer=save_answer)
#     eval.run_hist_method(N=iter_num)
#     eval.plot()

# %% [markdown]
# # A: Dual Eval

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
for mode in corr_lst:
    model_corr = mode if mode == None or mode == 'adjust' else f'{mode}_corr'
    print('=======================', model_corr, '=======================')
    eval = DrEval(filepaths, 'A', model_corr, sr_model_id, sr_model_dscp, dr_model_id, dr_model_dscp, dirc_mets=dirc_mets,
              anchor_mode=anchor_mode, test_mode=test_mode, save_answer=save_answer)
    eval.run_hist_method(N=iter_num)
    eval.plot()
