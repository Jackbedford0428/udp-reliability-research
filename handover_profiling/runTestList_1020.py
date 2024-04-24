from os.path import exists
import random
import subprocess
import multiprocessing
# For Progress Bar
# from tqdm import tqdm, trange
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
from time import time,strftime,gmtime
import pickle
from collections import Counter
import logging

test_scenario_default = [ 

    {
        'seed':[55688],
        'train':['train1','train2','train3'],#,'train'],
        'test':['test1','test2','test3'],
        # 'time_seq':[20],
        # 'predict_t':[2],
        # 'target':['setup','RLF'],
    }

    
    ]


class Run_Test_List:
    def expand(self,args:list,do_List:str) -> list:
        """將do_List作為檔案，把args依據排列組合添加在其後(有附上--以方便argsparser使用)

        Args:
            args (list): 需要添加的參數(添加會同時增加--在前面)
            do_List (str): 檔案名稱

        Returns:
            list: 添加完之後的list回傳
        """
        for i in args:
            preList, do_List = do_List, []
            for j in args[i]:
                do_List.extend([x + ' --' + i + ' ' + str(j) for x in preList])
        return do_List

    def runcmd(self,parameter:tuple) -> int:
        """額外建立子程序執行parameter

        Args:
            parameter (tuple(nice_number, command)): 將command建立子程序執行，並使用nice_number來設定nice值 

        Returns:
            int: 回傳執行結果
        """
        return_code = -1
        nice_number, command_org = parameter
        if self.nice_use:
            command = f'nice -n {nice_number % 10} {command_org}'
        else:
            command = command_org #不使用nice
        start_time = time()
        try:
            # r = subprocess.Popen(command ,shell=True, stdout=subprocess.DEVNULL) #不獲取stdout stderr
            r = subprocess.Popen(command ,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # return_code = r.wait()

            # 等待命令完成
            stdout, stderr = r.communicate()

            # 獲得命令的返回值
            return_code = r.returncode
            if return_code:
                # 如果任务执行失败，记录错误信息
                logging.getLogger('error').error(f"Task:'{command}' std:{str(stdout)}err:{str(stderr)}")
                # with open('debug.txt','a') as f_:
                #     f_.write(f'\n{command}')
                #     f_.write(f'\n std:{stdout}\n err:{stderr}')
                # raise RuntimeError("see debug")
                    
            else:
                logging.getLogger('success').info('command:' + command+'use time:' f'{strftime("%H:%M:%S", gmtime((time()- start_time)))}')
                # 标记任务为已完成
                logging.getLogger('completed_tasks').info(f"Marked task '{command_org}' as completed")
                # # 更新已完成清單
                # with open(self.completed_tasks_file, 'rb') as f:
                #         completed_tasks = pickle.load(f)
                # completed_tasks.add(command_org)
                # with open(self.completed_tasks_file, 'wb') as f:
                #         pickle.dump(completed_tasks, f)
        
        # except AssertionError as e:
        except Exception as e:
            # 如果任务执行失败，记录错误信息
            logging.error(f"Task '{command}' failed: {e}")
            # with open('notRun.txt','a') as f_:
            #     #  + 8 * 3600 是為了改成UTC+8
            #     f_.write(f'\ntime :{strftime("%Y-%m-%d, %H:%M:%S", gmtime((time() + 8 * 3600)))},{command},{e}')
            # return -1 if return_code is None else return_code
        
        
                
        return return_code


    # ---------- -----------------
    def __init__(self, file_py: str,/,model_list:list,*,test_scenario:list = None,cpu_count:int=15,ncols:int=70,nice_use:bool=True,completed_tasks_file:str=None):
        """初始化要測試的內容

        Args:
            file_py (str): 測試使用的py檔案
            model_list (list): model及其參數(多個)
            test_scenario (list): 測試環境參數(多個) # NOTE: have default 
            ncols (int, optional): 執行的進度條長度. Defaults to 50.
            cpu_count (int, optional): 設定一次要跑幾個. Defaults to 8.
        """
        # 設定一次跑幾個
        self.test_scenario = test_scenario_default if test_scenario is None else test_scenario
        self.count = cpu_count #multiprocessing.cpu_count()
        self.ncols = ncols # 執行的進度條長度
        self.nice_use = nice_use #是否使用nice
        self.file_py = file_py
        # 建立暫時存放已完成的列表
        self.completed_tasks_file = 'completed_tasks.log' if completed_tasks_file is None else completed_tasks_file

        # 將測試環境參數添加在測試用檔案之後
        temp = []
        for i in self.test_scenario:
            temp.extend(self.expand(i,[file_py]))
        doList = temp

        # 添加每個模型以及其對應的參數在測試用檔案之後
        temp = []
        for i in model_list:
            temp.extend(self.expand(i,doList))
        self.doList = temp

        # 创建文件处理器，用于记录已完成的任务
        completed_tasks_handler = logging.FileHandler('completed_tasks.log')
        logging.getLogger('completed_tasks').addHandler(completed_tasks_handler)
        logging.getLogger('completed_tasks').setLevel(logging.INFO)

        # 创建格式化器，包含时间信息
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 创建文件处理器，用于记录错误的日志
        error_handler = logging.FileHandler('error_logs.log')
        error_handler.setFormatter(formatter)
        logging.getLogger('error').addHandler(error_handler)
        logging.getLogger('error').setLevel(logging.ERROR) # 设置级别为 ERROR，记录 ERROR 及以上级别的日志


        # 创建文件处理器，用于记录成功和有瑕疵的日志
        success_and_defect_handler = logging.FileHandler('success_and_defect_logs.log')
        success_and_defect_handler.setFormatter(formatter)
        logging.getLogger('success').addHandler(success_and_defect_handler)
        logging.getLogger('success').setLevel(logging.INFO) # 设置级别为 INFO，记录 INFO 及以上级别的日志

        
    def loopCase(self):
        
        # with multiprocessing.Pool(processes=count) as p:
        #     list(tqdm(p.imap(self.runcmd, enumerate(self.doList)), total=len(self.doList), ncols=self.ncols))
        # with multiprocessing.Pool(processes=self.count) as p:
        #     r = list(p.imap(self.runcmd, tqdm(enumerate(self.doList), total=len(self.doList), ncols=self.ncols, desc=self.file_py.split('/')[-1])  ))

        r = process_map(self.runcmd, 
                        tqdm(enumerate(self.doList), total=len(self.doList), ncols=self.ncols, desc=self.file_py.split('/')[-1]), 
                        max_workers=self.count, total=len(self.doList), ncols=self.ncols, desc=self.file_py.split('/')[-1])
        # 使用 Counter 統計回傳值
        print ("done , follow is the return conuter")
        print (Counter(r))
        print('---------------')

    def run(self,Disable_complete_check:bool=False):
        # list_do = list1()
        print(f"{len(self.doList)} need run")
        # 读取已完成的任务集合
        completed_tasks = set()
        # 尝试加载已完成任务列表，如果不存在则创建一个空列表
        if not exists(self.completed_tasks_file):
            # completed_tasks = set()
            # with open(self.completed_tasks_file, 'wb') as f:
            #     pickle.dump(completed_tasks, f)
            pass
        else:
            # 从已完成任务的日志文件中提取已完成的任务
            with open('completed_tasks.log', 'r') as f:
                completed_tasks = set( line.split("'")[1] for line in f if "Marked task" in line)

        if Disable_complete_check is False:
            # 检查任务是否已经完成
            self.doList = [i for i in self.doList if i not in completed_tasks]
        
        print(f"{len(self.doList)} will run")
        # 使用 random.shuffle() 打亂model跑的順序，讓gpu和 cpu only交錯
        random.shuffle(self.doList)
        # print(list_do)
        self.loopCase()
        
