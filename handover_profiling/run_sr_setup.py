from runTestList import Run_Test_List
from pytictoc import TicToc

test_scenario = [
    {
        # '-m': ['dl_lost', 'dl_excl', 'ul_lost', 'ul_excl'],
        # '-r': ['BR', 'A', 'B', 'R', 'G', 'O2', 'subBR', 'subA', 'subB', 'subR', 'subG', 'subO2', 'all'],
        '-m': ['dl_lost'],
        # '-r': ['all'],
        # '-r': ['BR'],
        # '-dm': ['dl_lost'],
    }
]

t0 = TicToc()  # create instance of class
t0.tic()       # Start timer

# Run_Test_List('python3 ./setup_sr_handover_profile.py \
#     -it 3 -dt train \
#     -d 2024-03-19 \
#     -sm BR -r BR', test_scenario=test_scenario, cpu_count=10)

# Run_Test_List('python3 ./setup_sr_handover_profile.py \
#     -it 3 -dt train \
#     -d 2024-03-20 \
#     -sm A -r A', test_scenario=test_scenario, cpu_count=10)

# Run_Test_List('python3 ./setup_sr_handover_profile.py \
#     -it 3 -dt train \
#     -d 2024-06-18-1 \
#     -sm B -r B', test_scenario=test_scenario, cpu_count=10),

# Run_Test_List('python3 ./setup_sr_handover_profile.py \
#     -it 3 -dt train \
#     -d 2024-06-19-1 \
#     -sm G -r G', test_scenario=test_scenario, cpu_count=10)

# Run_Test_List('python3 ./setup_sr_handover_profile.py \
#     -it 3 -dt train \
#     -d 2024-06-20-1 \
#     -sm R -r R', test_scenario=test_scenario, cpu_count=10)

# Run_Test_List('python3 ./setup_sr_handover_profile.py \
#     -it 3 -dt train \
#     -d 2024-06-21-1 \
#     -sm O2 -r O2', test_scenario=test_scenario, cpu_count=10)

# Run_Test_List('python3 ./setup_sr_handover_profile.py \
#     -it 3 -dt train -sm subBR -r subBR \
#     -d 2024-03-20 2024-06-18-1 2024-06-19-1 2024-06-20-1 2024-06-21-1',
#     test_scenario=test_scenario, cpu_count=10)

# Run_Test_List('python3 ./setup_sr_handover_profile.py \
#     -it 3 -dt train -sm subA -r subA \
#     -d 2024-03-19 2024-06-18-1 2024-06-19-1 2024-06-20-1 2024-06-21-1',
#     test_scenario=test_scenario, cpu_count=10)

# Run_Test_List('python3 ./setup_sr_handover_profile.py \
#     -it 3 -dt train -sm subB -r subB \
#     -d 2024-03-19 2024-03-20 2024-06-19-1 2024-06-20-1 2024-06-21-1',
#     test_scenario=test_scenario, cpu_count=10)

# Run_Test_List('python3 ./setup_sr_handover_profile.py \
#     -it 3 -dt train -sm subG -r subG \
#     -d 2024-03-19 2024-03-20 2024-06-18-1 2024-06-20-1 2024-06-21-1',
#     test_scenario=test_scenario, cpu_count=10)

# Run_Test_List('python3 ./setup_sr_handover_profile.py \
#     -it 3 -dt train -sm subR -r subR \
#     -d 2024-03-19 2024-03-20 2024-06-18-1 2024-06-19-1 2024-06-21-1',
#     test_scenario=test_scenario, cpu_count=10)

# Run_Test_List('python3 ./setup_sr_handover_profile.py \
#     -it 3 -dt train -sm subO2 -r subO2 \
#     -d 2024-03-19 2024-03-20 2024-06-18-1 2024-06-19-1 2024-06-20-1',
#     test_scenario=test_scenario, cpu_count=10)

Run_Test_List('python3 ./setup_sr_handover_profile.py \
    -it 3 -dt train -sm all -r all \
    -d 2024-03-19 2024-03-20 2024-06-18-1 2024-06-19-1 2024-06-20-1 2024-06-21-1',
    test_scenario=test_scenario, cpu_count=10)

# Run_Test_List('python3 ./test.py')

t0.toc()  # Time elapsed since t.tic()
