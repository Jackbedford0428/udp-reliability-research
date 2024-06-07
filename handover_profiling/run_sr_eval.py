from runTestList_1020 import Run_Test_List
from pytictoc import TicToc

test_scenario = [
    {
        '-dm': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
        '-r': ['BR', 'A'],
        # '-r': ['BR'],
        # '-dm': ['dl_lost'],
    }
]

t0 = TicToc()  # create instance of class
t0.tic()       # Start timer

Run_Test_List('python3 ./eval_sr_handover_profiling_param.py \
    -sa -it 5 -dt test \
    -d 2024-05-14 2024-05-15 \
    -m 20240417_1333007d66_new_data_sync_v2', test_scenario=test_scenario, cpu_count=10)

# Run_Test_List('python3 ./test.py')

t0.toc()  # Time elapsed since t.tic()
