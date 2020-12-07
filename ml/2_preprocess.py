import os, sys, json
import logging as log
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

lines = sys.stdin.readlines()
user_wifi = json.loads(lines[0])
pos_name, lat, lon = sys.argv[1:4]

rssi_threshold = 40

df_pos = pd.DataFrame.from_records(user_wifi)
df_pos = df_pos[df_pos['rssi'] > rssi_threshold]

log_path = Path(__file__).parent / 'log'
log_path.mkdir(parents=True, exist_ok=True)

handler = log.FileHandler(log_path / 'preprocess.log', 'a+', 'utf-8')
log.basicConfig(handlers=[handler], level=log.INFO)

cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
# 전체 Wi-Fi 데이터 목록에서 rp 별로 분리
for rp_name, df_rp in df_pos.groupby('rp'):
    rp_path = Path(__file__).parent / 'data' / pos_name / rp_name
    rp_path.mkdir(parents=True, exist_ok=True)

    # defaultdict.fromkeys()에서 알아서 중복 제거해줌
    # key = timestamp, value = defaultdict of bssid in rp : rssi, rp : no. of rp
    dict_rp = defaultdict(lambda : defaultdict(int).fromkeys(df_rp['bssid'], 0))
    for idx, wifi in df_rp.iterrows():
        dict_rp[wifi['timestamp']][wifi['bssid']] = wifi['rssi']
        dict_rp[wifi['timestamp']]['rp'] = wifi['rp']

    # column name으로 bssid, rp를 남겨둬서 모델 생성 시 활용
    train_data = pd.DataFrame.from_dict(dict_rp).transpose()

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_data.to_csv(rp_path / f'{cur_time}.csv', index=False, encoding="utf-8")
    log.info(f'{datetime.now()} train data generated for {pos_name} {rp_name} {cur_time}')



