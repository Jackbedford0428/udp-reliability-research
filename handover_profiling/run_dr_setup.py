from runTestList import Run_Test_List
from pytictoc import TicToc


# %%
t = TicToc()  # create instance of class
t.tic()       # Start timer

test_scenario = [
    {
        # '-r': ['BR', 'A', 'B', 'R', 'G', 'O2', 'subBR', 'subA', 'subB', 'subR', 'subG', 'subO2', 'all'],
        # '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
        # '-am': ['by_event', 'by_packet'],
        '-r': ['BR'],
        '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
        '-am': ['by_event', 'by_packet'],
    }
]

Run_Test_List('python3 ./setup_dr_handover_profile.py -n1 20240824_example -t',
              test_scenario=test_scenario, cpu_count=15)

t.toc()  # Time elapsed since t.tic()
