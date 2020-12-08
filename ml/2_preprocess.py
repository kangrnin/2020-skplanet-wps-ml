import os, sys, json
import logging as log
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

lines = sys.stdin.readlines()
scan_wifi = json.loads(lines[0])
pos_name, lat, lon = sys.argv[1:4]

df_pos = pd.DataFrame.from_records(scan_wifi)

(Path(__file__).parent / '../log').mkdir(parents=True, exist_ok=True)
handler = log.FileHandler(Path(__file__).parent / '../log/preprocess.log', 'a+', 'utf-8')
log.basicConfig(handlers=[handler], level=log.INFO)

cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")

for rp_name, df_rp in df_pos.groupby('rp'):
    rp_path = Path(__file__).parent / '../train_data' / pos_name / rp_name
    rp_path.mkdir(parents=True, exist_ok=True)

    dict_rp = defaultdict(lambda : defaultdict(int).fromkeys(df_rp['bssid'], 0))
    for idx, wifi in df_rp.iterrows():
        dict_rp[wifi['timestamp']][wifi['bssid']] = wifi['rssi']
        dict_rp[wifi['timestamp']]['rp'] = wifi['rp']

    train_data = pd.DataFrame.from_dict(dict_rp).transpose()
    train_data.to_csv(rp_path / f'{cur_time}.csv', index=False, encoding="utf-8")
    log.info(f'{datetime.now()} train data generated for {pos_name} {rp_name} {cur_time}')