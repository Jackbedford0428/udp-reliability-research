from runTestList import Run_Test_List
from pytictoc import TicToc

# %%
### Single Route -- Training Set

t = TicToc()  # create instance of class
t.tic()       # Start timer

test_scenario = [
    {
        # '-r': ['BR', 'A', 'B', 'R', 'G', 'O2'],
        # '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
        '-r': ['BR'],
        '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
    }
]

Run_Test_List('python3 ./eval_sr_handover_profile.py -n1 20240824_example -it 1 -du training -t',
              test_scenario=test_scenario, cpu_count=15)

t.toc()  # Time elapsed since t.tic()


# %%
### Single Route -- Testing Set

t = TicToc()  # create instance of class
t.tic()       # Start timer

test_scenario = [
    {
        # '-r': ['BR', 'A', 'B', 'R', 'G', 'O2'],
        # '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
        '-r': ['BR'],
        '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
    }
]

Run_Test_List('python3 ./eval_sr_handover_profile.py -n1 20240824_example -it 1 -du testing -t',
              test_scenario=test_scenario, cpu_count=15)

t.toc()  # Time elapsed since t.tic()


# # %%
# ### Subtract Route -- Testing Set

# t = TicToc()  # create instance of class
# t.tic()       # Start timer

# test_scenario = [
#     {
#         # '-r': ['subBR', 'subA', 'subB', 'subR', 'subG', 'subO2'],
#         # '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
#         '-r': ['subBR'],
#         '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
#     }
# ]

# Run_Test_List('python3 ./eval_sr_handover_profile.py -n1 20240824_example -it 1 -du testing -t',
#               test_scenario=test_scenario, cpu_count=15)

# t.toc()  # Time elapsed since t.tic()


# # %%
# ### All Routes -- Training Set

# t = TicToc()  # create instance of class
# t.tic()       # Start timer

# test_scenario = [
#     {
#         # '-r': ['BR', 'A', 'B', 'R', 'G', 'O2', 'all'],
#         # '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
#         '-r': ['BR'],
#         '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
#     }
# ]

# Run_Test_List('python3 ./eval_sr_handover_profile.py -n1 20240824_example -n2 all -it 1 -du training -t',
#               test_scenario=test_scenario, cpu_count=15)

# t.toc()  # Time elapsed since t.tic()


# # %%
# ### All Routes -- Testing Set

# t = TicToc()  # create instance of class
# t.tic()       # Start timer

# test_scenario = [
#     {
#         # '-r': ['BR', 'A', 'B', 'R', 'G', 'O2', 'all'],
#         # '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
#         '-r': ['BR'],
#         '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
#     }
# ]

# Run_Test_List('python3 ./eval_sr_handover_profile.py -n1 20240824_example -n2 all -it 1 -du testing -t',
#               test_scenario=test_scenario, cpu_count=15)

# t.toc()  # Time elapsed since t.tic()
