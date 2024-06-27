from runTestList_1020 import Run_Test_List
from pytictoc import TicToc

test_scenario = [
    {
        # '-am': ['by_event', 'by_packet'],
        '-am': ['by_event'],
        '-cc': [None, 'adjust', 'zero', 'max', '25%', '50%', '75%'],
        '-r': ['BR', 'A'],
        '-dm': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
        # '-r': ['BR'],
        # '-dm': ['dl_lost'],
        # '-am': ['by_event'],
        # '-cc': [None]
    }
]

t0 = TicToc()  # create instance of class
t0.tic()       # Start timer

Run_Test_List('python3 ./eval_dr_handover_profiling_param.py \
    -sa -it 1 -dt test \
    -d 2024-05-14 2024-05-15 \
    -srm 20240417_1333007d66_new_data_sync_v2 \
    -drm dr_20240417_1452001e84', test_scenario=test_scenario, cpu_count=20)

# Run_Test_List('python3 ./test.py')

t0.toc()  # Time elapsed since t.tic()
