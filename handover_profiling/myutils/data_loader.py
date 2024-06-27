#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import itertools as it
import yaml
from pprint import pprint

__all__ = [
    "data_loader",
    "set_data",
    "data_aligner",
    "data_consolidator",
]

with open(os.path.join(os.path.dirname(__file__), "db_path.txt"), "r") as f:
    PATH_TO_DATABASE = f.readline()

def data_loader(
    mode='sr', query_dates=False, show_info=False,
    selected_dates=[], selected_exps=[], selected_routes=[],
    excluded_dates=[], excluded_exps=[], excluded_routes=[],
    root_dir=PATH_TO_DATABASE):
    
    # Collect experiment dates
    dates = [s for s in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, s)) and s not in ['backup']]
    
    if query_dates:
        if show_info:
            star_flag = True
            date_dirs = [os.path.join(root_dir, s) for s in dates]
            for date, date_dir in zip(dates, date_dirs):
                date = os.path.basename(date_dir)
                
                # Specify path to JSON file
                json_filepath = os.path.join(date_dir, f'{date}.json')
                
                if not os.path.isfile(json_filepath):
                    if star_flag:
                        print('*************************************************************************************')
                        star_flag = False
                    print(f'{json_filepath} does not exist!!!')
                    print('*************************************************************************************')
                    continue
                else:
                    star_flag = True
                
                # Read the JSON file and load its contents into a dictionary
                with open(json_filepath, 'r', encoding='utf-8') as json_file:
                    my_dict = json.load(json_file)
                
                # If the JSON file is empty, then continue
                if not my_dict:
                    continue
                
                print(date, len(my_dict))
                for exp, item in my_dict.items():
                    print({exp: item})
                
        return dates
    
    # Collect experiments
    date_dirs = [os.path.join(root_dir, s) for s in selected_dates if s not in excluded_dates]
    exps_dict = {}
    
    for date_dir in date_dirs:
        date = os.path.basename(date_dir)
        
        try:
            yaml_filepath = os.path.join(date_dir, f'{date}.yml')
            with open(yaml_filepath, 'r', encoding='utf-8') as yaml_file:
                my_dict = yaml.safe_load(yaml_file)
            # pprint(my_dict)
        
        except:
            # Specify path to JSON file
            json_filepath = os.path.join(date_dir, f'{date}.json')
            
            # Read the JSON file and load its contents into a dictionary
            with open(json_filepath, 'r', encoding='utf-8') as json_file:
                my_dict = json.load(json_file)
        
        # If the JSON file is empty, then continue
        if not my_dict:
            continue
        
        for i, (exp, item) in enumerate(my_dict.items()):
            if len(selected_exps) != 0 and exp not in selected_exps:
                continue
            if len(excluded_exps) != 0 and exp in excluded_exps:
                continue
            if len(selected_routes) != 0 and item['route'] not in selected_routes:
                continue
            if len(excluded_routes) != 0 and item['route'] in excluded_routes:
                continue
            try:
                exps_dict[date] = {**exps_dict[date], **{exp: item}}
            except:
                exps_dict[date] = {exp: item}
    
    if show_info:            
        for date, exps in exps_dict.items():
            print(date, len(exps))
            for exp_name, exp in exps.items():
                print({exp_name: exp})
    
    filepaths = []
    if mode == 'sr':
        for date, exps in exps_dict.items():
            # print(date, len(exps))
            
            for exp_name, exp in exps.items():
                exp_dir = os.path.join(root_dir, date, exp_name)
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
                exp_dir = os.path.join(root_dir, date, exp_name)
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
