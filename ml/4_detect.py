import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from wifi_scan_bssid import get_wifis
from collections import defaultdict

position = sys.argv[1]
model_path = Path('../ml_model') / position
model_rdf = joblib.load(model_path / 'model_rdf.plk')
model_svm = joblib.load(model_path / 'model_svm.plk')

rps = np.load(model_path / 'classes.npy')
bssids = np.load(model_path / 'features.npy')

wifis = get_wifis()

wifi_dict = defaultdict.fromkeys(wifi_df.columns, 0)
for wifi in wifis:
    if wifi['bssid'] in wifi_dict:
        wifi_dict[wifi['bssid']] = int(wifi['signal'][:-1])

print('current location : '+str(model.predict(wifi_dict)[0]))