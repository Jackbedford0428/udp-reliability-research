from runTestList import Run_Test_List
from pytictoc import TicToc


# # Training Set - Single
# t = TicToc()  # create instance of class
# t.tic()       # Start timer

# test_scenario = [
#     {
#         '-cc': ['mle', 'adjust', 'zero', 'max', '25%', '50%', '75%'],
#         '-mm': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
#         '-r': ['BR', 'A', 'B', 'R', 'G', 'O2'],
#         # '-m': ['subBR', 'subA', 'subB', 'subR', 'subG', 'subO2'],
#     }
# ]

# Run_Test_List('python3 ./eval_dr_handover_profile.py \
#     -dt train -am by_event -it 1 \
#     -p 20240630_latest_model \
#     -d 2024-03-19 2024-03-20 2024-06-18-1 2024-06-19-1 2024-06-20-1 2024-06-21-1',
#     test_scenario=test_scenario, cpu_count=30)

# t.toc()  # Time elapsed since t.tic()


# # Testing Set - Single
# t = TicToc()  # create instance of class
# t.tic()       # Start timer

# test_scenario = [
#     {
#         '-cc': ['mle', 'adjust', 'zero', 'max', '25%', '50%', '75%'],
#         '-mm': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
#         '-r': ['BR', 'A', 'B', 'R', 'G', 'O2'],
#         # '-m': ['subBR', 'subA', 'subB', 'subR', 'subG', 'subO2'],
#     }
# ]

# Run_Test_List('python3 ./eval_dr_handover_profile.py \
#     -dt test -am by_event -it 1 \
#     -p 20240630_latest_model \
#     -d 2024-05-14 2024-05-15 2024-06-18-2 2024-06-19-2 2024-06-20-2 2024-06-21-2',
#     test_scenario=test_scenario, cpu_count=30)

# t.toc()  # Time elapsed since t.tic()


# Testing Set - Sub
t = TicToc()  # create instance of class
t.tic()       # Start timer

test_scenario = [
    {
        '-cc': ['mle', 'adjust', 'zero', 'max', '25%', '50%', '75%'],
        '-mm': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
        # '-r': ['BR', 'A', 'B', 'R', 'G', 'O2'],
        # '-m': ['subBR', 'subA', 'subB', 'subR', 'subG', 'subO2'],
        '-m': ['subA', 'subB', 'subR', 'subG', 'subO2'],
    }
]

Run_Test_List('python3 ./eval_dr_handover_profile.py \
    -dt test -am by_event -it 1 \
    -p 20240630_latest_model \
    -d 2024-05-14 2024-05-15 2024-06-18-2 2024-06-19-2 2024-06-20-2 2024-06-21-2',
    test_scenario=test_scenario, cpu_count=25)

t.toc()  # Time elapsed since t.tic()

# python3 ./eval_dr_handover_profile.py -dt test -am by_event -it 1 -p 20240630_latest_model_test -d 2024-05-14 2024-05-15 2024-06-18-2 2024-06-19-2 2024-06-20-2 2024-06-21-2 -cc mle -mm dl_lost -m subA


# # Training Set - All
# t = TicToc()  # create instance of class
# t.tic()       # Start timer

# test_scenario = [
#     {
#         '-cc': ['mle', 'adjust', 'zero', 'max', '25%', '50%', '75%'],
#         '-mm': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
#         '-r': ['BR', 'A', 'B', 'R', 'G', 'O2'],
#         # '-m': ['subBR', 'subA', 'subB', 'subR', 'subG', 'subO2'],
#     }
# ]

# Run_Test_List('python3 ./eval_dr_handover_profile.py \
#     -m all -dt train -am by_event -it 1 \
#     -p 20240630_latest_model \
#     -d 2024-03-19 2024-03-20 2024-06-18-1 2024-06-19-1 2024-06-20-1 2024-06-21-1',
#     test_scenario=test_scenario, cpu_count=30)

# t.toc()  # Time elapsed since t.tic()


# # Testing Set - All
# t = TicToc()  # create instance of class
# t.tic()       # Start timer

# test_scenario = [
#     {
#         '-cc': ['mle', 'adjust', 'zero', 'max', '25%', '50%', '75%'],
#         '-mm': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
#         '-r': ['BR', 'A', 'B', 'R', 'G', 'O2'],
#         # '-m': ['subBR', 'subA', 'subB', 'subR', 'subG', 'subO2'],
#     }
# ]

# Run_Test_List('python3 ./eval_dr_handover_profile.py \
#     -m all -dt test -am by_event -it 1 \
#     -p 20240630_latest_model \
#     -d 2024-05-14 2024-05-15 2024-06-18-2 2024-06-19-2 2024-06-20-2 2024-06-21-2',
#     test_scenario=test_scenario, cpu_count=30)

# t.toc()  # Time elapsed since t.tic()
