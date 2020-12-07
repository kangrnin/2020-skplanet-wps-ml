from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from datetime import datetime
import logging as log

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

lines = sys.stdin.readlines()
position = sys.argv[1]

log_path = Path('../log') / position
log_path.mkdir(parents=True, exist_ok=True)
handler = log.FileHandler(log_path / 'detect.log', 'a+', 'utf-8')
log.basicConfig(handlers=[handler], level=log.INFO)
log.info(f'{datetime.now()} ------- detection for {position} -------')

model_path = Path('../ml_model') / position
model_rdf = joblib.load(model_path / 'model_rdf.plk')
model_svm = joblib.load(model_path / 'model_svm.plk')

bssids = np.load(model_path / 'features.npy', allow_pickle=True)

rp_encoder = LabelEncoder()
rp_encoder.classes_ = np.load(model_path / 'classes.npy', allow_pickle=True)

user_wifis = json.loads(lines[0])
log.info(f'user data bssids total {len(user_wifis)}')

wifi_dict = defaultdict.fromkeys(bssids, 0)
for user_wifi in user_wifis:
    if user_wifi['bssid'] in wifi_dict:
        wifi_dict[user_wifi['bssid']] = user_wifi['rssi']

log.info(' '.join(np.asarray([rssi for bssid, rssi in wifi_dict.items()])))

input_df = pd.DataFrame.from_dict([dict(wifi_dict)])
input_df

pred_rdf = model_rdf.predict(input_df)
pred_svm = model_svm.predict(input_df)

rdf_rp = rp_encoder.inverse_transform(prediction)[0]
svm_rp = rp_encoder.inverse_transform(prediction)[0]

rdf_pred = model_rdf.predict_proba(input_df)[0]
svm_pred = model_svm.predict_proba(input_df)[0]

log.info(f'rdf : ')
log.info(', '.join(sorted(list(zip(rdf_pred, rp_encoder.classes_)), key=lambda p:p[0], reverse=True)))
log.info(f'svm : ')
log.info(', '.join(sorted(list(zip(svm_pred, rp_encoder.classes_)), key=lambda p:p[0], reverse=True)))

ave_pred = [(p1+p2)/2 for p1,p2 in list(zip(rdf_pred, svm_pred))]
ave_pred_rp = list(zip(ave_pred, rp_encoder.classes_))
ave_pred_rp_sorted = sorted(ave_pred_rp, key=lambda p: p[0], reverse=True)
log.info(f'ave : ')
log.info(', '.join(ave_pred_rp_sorted))