#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import itertools as it
import yaml
from pprint import pprint
from collections import defaultdict

__all__ = [
    "data_loader",
    "set_data",
    "data_aligner",
    "data_consolidator",
]

with open(os.path.join(os.path.dirname(__file__), "db_path.txt"), "r") as f:
    PATHS_TO_DATABASE = [s.strip() for s in f.readlines()]

def data_loader(
    mode='sr', query_dates=False, verbose=False,
    ue = 'any',
    selected_dates=[], selected_exps=[], selected_routes=[],
    excluded_dates=[], excluded_exps=[], excluded_routes=[],
    root_paths=PATHS_TO_DATABASE):
    
    # Collect experiment dates
    if query_dates:
        
        dates = []
        dates_verbose_lst = []
        
        for root_path in root_paths:
            dates = dates + [s for s in sorted(os.listdir(root_path)) if os.path.isdir(os.path.join(root_path, s))]
            dates_verbose_lst = dates_verbose_lst + [[s, root_path] for s in sorted(os.listdir(root_path)) if os.path.isdir(os.path.join(root_path, s))]
        
        unique_dates = sorted(list(set(dates)))
        
        date_dict = defaultdict(set)  # 使用 set 來自動去除重複的資料夾

        for date, folder in dates_verbose_lst:
            date_dict[date].add(folder)

        # 如果你需要將 set 轉換回 list (可選，視你需求而定)
        dates_verbose = {key: list(value) for key, value in date_dict.items()}
        dates_verbose = dict(sorted(dates_verbose.items(), key=lambda x: x[0]))
        
        if verbose:
            for date, folders in dates_verbose.items():
                if len(folders) > 1:
                    print(date, end=' ')
                    for folder in folders:
                        print(folder, end=' ')
                    print()
                else:
                    print(date, folders[0])
                    
        return unique_dates

    # Collect experiments
    date_paths = []
    
    for date in selected_dates:
        not_found = []
        
        if date in excluded_dates:
            continue
        
        for root_path in root_paths:
            date_path = os.path.join(root_path, date)
            if os.path.isdir(date_path):
                date_paths.append(date_path)
            else:
                not_found.append(date_path)
        
        if len(not_found) == len(root_paths):
            error_message = "[Errno 2] No such file or directory:\n"
            for date_path in not_found:
                error_message += "  '{}'\n".format(date_path)
            raise FileNotFoundError(error_message.strip())
    
    exps_dict = {}
    
    for date_path in date_paths:
        if verbose:
            print('------------------')
            print(os.path.basename(date_path), os.path.dirname(date_path))
        
        yaml_filepath = os.path.join(date_path, os.path.basename(date_path) + '.yml')
        with open(yaml_filepath, 'r', encoding='utf-8') as yaml_file:
            my_dict = yaml.safe_load(yaml_file)
        
        # If the YAML file is empty, then continue
        if not my_dict:
            continue
        
        date = os.path.basename(date_path)
        
        for exp_name, exp in my_dict.items():
            if verbose:
                print(' ', exp_name, '->', 'Skip:', exp['skip'], '|', 'UE:', exp['ue'], '|', 'Laptop:', exp['laptop'], '|', 'Route:', exp['route'])
                print('   ', exp['devices'])
                
            if ue == 'any':
                if exp['skip']:
                    continue
            else:
                if exp['skip'] and exp['ue'] != ue:  # Phone or Modem
                    continue
            
            if len(selected_exps) != 0 and exp_name not in selected_exps:
                continue
            if len(excluded_exps) != 0 and exp_name in excluded_exps:
                continue
            if len(selected_routes) != 0 and exp['route'] not in selected_routes:
                continue
            if len(excluded_routes) != 0 and exp['route'] in excluded_routes:
                continue
            try:
                exps_dict[date] = {**exps_dict[date], **{exp_name: exp}}
            except:
                exps_dict[date] = {exp_name: exp}
            
            if verbose:
                for dev in exp['devices']:
                    print('   ', dev)
                    
                    for trip in exp['ods']:
                        if trip == 0:
                            continue
                        else:
                            trip = f'#{trip:02d}'
                        data_dir = os.path.join(date_path, exp_name, dev, trip)
                        print('     ', data_dir, os.path.isdir(data_dir))
    
    filepaths = []
    if mode == 'sr':
        for date, exps in exps_dict.items():
            # print(date, len(exps))
            
            for exp_name, exp in exps.items():
                exp_dir = os.path.join(root_path, date, exp_name)
                # print({exp_name: exp})
                
                devices = list(exp['devices'].keys())
                try:
                    trips = ['#{:02d}'.format(s[0]) for s in exp['ods'][1:]]
                except:
                    trips = ['#{:02d}'.format(s) for s in list(exp['ods'].keys())[1:]]
                for trip in trips:
                    for dev in devices:
                        data_dir = os.path.join(exp_dir, dev, trip, 'data')
                        filepaths.append([
                            os.path.join(data_dir, 'handover_info_log.csv'),
                            os.path.join(data_dir, 'udp_dnlk_loss_latency.csv'),
                            os.path.join(data_dir, 'udp_uplk_loss_latency.csv'),
                            os.path.join(data_dir, [s for s in os.listdir(data_dir) if s.endswith('rrc.csv')][0]),
                            os.path.join(data_dir, [s for s in os.listdir(data_dir) if s.endswith('ml1.csv') and not s.endswith('nr_ml1.csv')][0]),
                            os.path.join(data_dir, [s for s in os.listdir(data_dir) if s.endswith('nr_ml1.csv')][0]),
                            ])
    elif mode == 'dr':
        for date, exps in exps_dict.items():
            # print(date, len(exps))
            
            for exp_name, exp in exps.items():
                exp_dir = os.path.join(root_path, date, exp_name)
                # print({exp_name: exp})
                
                devices = list(exp['devices'].keys())
                combos = list(it.combinations(devices, 2))
                try:
                    trips = ['#{:02d}'.format(s[0]) for s in exp['ods'][1:]]
                except:
                    trips = ['#{:02d}'.format(s) for s in list(exp['ods'].keys())[1:]]
                for trip in trips:
                    for dev1, dev2 in combos:
                        data_dir1 = os.path.join(exp_dir, dev1, trip, 'data')
                        data_dir2 = os.path.join(exp_dir, dev2, trip, 'data')
                        _filepaths = []
                        for i in range(2):
                            _filepaths.append([
                                os.path.join(locals()[f'data_dir{i+1}'], 'handover_info_log.csv'),
                                os.path.join(locals()[f'data_dir{i+1}'], 'udp_dnlk_loss_latency.csv'),
                                os.path.join(locals()[f'data_dir{i+1}'], 'udp_uplk_loss_latency.csv'),
                                os.path.join(locals()[f'data_dir{i+1}'], [s for s in os.listdir(locals()[f'data_dir{i+1}']) if s.endswith('rrc.csv')][0]),
                                os.path.join(locals()[f'data_dir{i+1}'], [s for s in os.listdir(locals()[f'data_dir{i+1}']) if s.endswith('ml1.csv') and not s.endswith('nr_ml1.csv')][0]),
                                os.path.join(locals()[f'data_dir{i+1}'], [s for s in os.listdir(locals()[f'data_dir{i+1}']) if s.endswith('nr_ml1.csv')][0]),
                            ])
                        filepaths.append(tuple(_filepaths))
    return filepaths

def set_data(df, mode='pcap', tz=0):
    def nr_serv_cel(row):
        pos = row.serv_cel_pos
        if pos == 255:
            return 65535, -160, -50
        else:
            return row[f'PCI{pos}'], row[f'RSRP{pos}'], row[f'RSRQ{pos}']
    
    if mode == 'pcap':
        common_column_names = ['seq', 'rpkg', 'frame_id', 'Timestamp', 'lost', 'excl', 'latency', 'xmit_time', 'arr_time']
        
        if df.empty:
            return pd.DataFrame(columns=common_column_names)
        
        date_columns = ['Timestamp', 'xmit_time', 'arr_time']
        df[date_columns] = df[date_columns].apply(pd.to_datetime)
        df[['seq', 'rpkg', 'frame_id']] = df[['seq', 'rpkg', 'frame_id']].astype('Int32')
        df[['latency']] = df[['latency']].astype('float32')
        df[['lost', 'excl']] = df[['lost', 'excl']].astype('boolean')

    if mode in ['lte', 'nr']:
        common_column_names = [
            'Timestamp', 'type_id', 'PCI', 'RSRP', 'RSRQ', 'serv_cel_index', 'EARFCN', 'NR_ARFCN', 
            'num_cels', 'num_neigh_cels', 'serv_cel_pos', 'PCI0', 'RSRP0', 'RSRQ0',
        ]
        
        if df.empty:
            return pd.DataFrame(columns=common_column_names)
        
        if mode == 'lte':
            columns_mapping = {
                'RSRP(dBm)': 'RSRP',
                'RSRQ(dB)': 'RSRQ',
                'Serving Cell Index': 'serv_cel_index',
                'Number of Neighbor Cells': 'num_neigh_cels',
                'Number of Detected Cells': 'num_cels',
            }
            columns_order = [*common_column_names, *df.columns[df.columns.get_loc('PCI1'):].tolist()]
            
            df = df.rename(columns=columns_mapping).reindex(columns_order, axis=1)
            df['serv_cel_index'] = np.where(df['serv_cel_index'] == '(MI)Unknown', '3_SCell', df['serv_cel_index'])
            df['num_cels'] = df['num_neigh_cels'] + 1
            df['type_id'] = 'LTE_PHY'

        if mode == 'nr':
            columns_mapping = {
                'Raster ARFCN': 'NR_ARFCN',
                'Serving Cell Index': 'serv_cel_pos',
                'Num Cells': 'num_cels',
            }
            columns_order = [*common_column_names, *df.columns[df.columns.get_loc('PCI1'):].tolist()]
            
            df = df.rename(columns=columns_mapping).reindex(columns_order, axis=1)
            df[['PCI', 'RSRP', 'RSRQ']] = df.apply(nr_serv_cel, axis=1, result_type='expand')
            df['serv_cel_index'] = np.where(df['serv_cel_pos'] == 255, df['serv_cel_index'], 'PSCell')
            df['num_neigh_cels'] = np.where(df['serv_cel_pos'] == 255, df['num_cels'], df['num_cels'] - 1)
            df['type_id'] = '5G_NR_ML1'
        
        df['Timestamp'] = pd.to_datetime(df['Timestamp']) + pd.Timedelta(hours=tz)
        df[['type_id', 'serv_cel_index']] = df[['type_id', 'serv_cel_index']].astype('category')
        df[['EARFCN', 'NR_ARFCN']] = df[['EARFCN', 'NR_ARFCN']].astype('Int32')
        df[['num_cels', 'num_neigh_cels', 'serv_cel_pos']] = df[['num_cels', 'num_neigh_cels', 'serv_cel_pos']].astype('UInt8')

        for tag in df.columns:
            if tag.startswith('PCI'):
                df[tag] = df[tag].astype('Int32')
            if tag.startswith(('RSRP', 'RSRQ')):
                df[tag] = df[tag].astype('float32')

    return df

def data_aligner(df, ho_df):
    empty_data = False
    
    if df.empty or ho_df.empty:
        empty_data = True
        return df, ho_df, empty_data
    
    start_ts = df.iloc[0]['Timestamp'] - pd.Timedelta(seconds=1)
    end_ts = df.iloc[-1]['Timestamp'] + pd.Timedelta(seconds=1)
    ho_df = ho_df[(ho_df['start'] >= start_ts) & (ho_df['start'] < end_ts)].reset_index(drop=True)
    
    if ho_df.empty:
        empty_data = True
        return df, ho_df, empty_data
    
    start_ts = ho_df.iloc[0]['start'] - pd.Timedelta(seconds=100)
    end_ts = ho_df.iloc[-1]['start'] + pd.Timedelta(seconds=100)
    df = df[(df['Timestamp'] >= start_ts) & (df['Timestamp'] < end_ts)].reset_index(drop=True)
    
    return df, ho_df, empty_data

def data_consolidator(df1, df2, ho_df1, ho_df2):
    empty_data = False
    
    df1, ho_df1, empty_data1 = data_aligner(df1, ho_df1)
    df2, ho_df2, empty_data2 = data_aligner(df2, ho_df2)
    if empty_data1 or empty_data2:
        empty_data = True
        return None, df1, df2, ho_df1, ho_df2, empty_data
    
    df = pd.merge(df1, df2, on='seq', how='inner').reset_index(drop=True)
    df1 = df[['seq', 'Timestamp_x', 'lost_x', 'excl_x', 'latency_x']].rename(columns={'Timestamp_x': 'Timestamp', 'lost_x': 'lost', 'excl_x': 'excl', 'latency_x': 'latency'})
    df2 = df[['seq', 'Timestamp_y', 'lost_y', 'excl_y', 'latency_y']].rename(columns={'Timestamp_y': 'Timestamp', 'lost_y': 'lost', 'excl_y': 'excl', 'latency_y': 'latency'})
    df1, ho_df1, _ = data_aligner(df1, ho_df1)
    df2, ho_df2, _ = data_aligner(df2, ho_df2)
    
    return df, df1, df2, ho_df1, ho_df2, empty_data
