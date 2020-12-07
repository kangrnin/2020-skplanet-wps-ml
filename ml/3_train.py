import os, sys, json
import logging as log
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

pos_name = sys.argv[1]

base_path = Path(__file__).parent
data_path = base_path / 'data' / pos_name
model_path = base_path / 'model' / pos_name
data_path.mkdir(parents=True, exist_ok=True)
model_path.mkdir(parents=True, exist_ok=True)

handler = log.FileHandler(base_path / 'log/train.log', 'a+', 'utf-8')
log.basicConfig(handlers=[handler], level=log.INFO)
log.info(f'{datetime.now()} ------- generating model for {pos_name} -------')

# data_path에 있는 모든 데이터파일을 dataframe으로 불러와 합치기 + 없는 값(NaN)을 0으로 채우기
df_all = pd.concat([pd.read_csv(path, encoding='utf-8') for path in data_path.glob('**/*') if path.is_file()]).fillna(0)
# 'rp' column을 맨 뒤로 보내기
df_all = df_all[[col for col in df_all.columns if col != 'rp'] + ['rp']]
for rp in np.unique(df_all['rp']):
    log.info(f'{datetime.now()} [{rp}] scan cnt : {len(df_all[df_all["rp"] == rp])}')

rp_encoder = LabelEncoder()
rp_encoder.fit(np.unique(df_all['rp']))

df_all['rp'] = rp_encoder.transform(df_all['rp'])

np.save(model_path / 'classes.npy', rp_encoder.classes_)

X = df_all.iloc[:,:-1].values
y = df_all.iloc[:,-1].values

kf = KFold(n_splits=5, shuffle=True, random_state=12321)

log.info(f'{datetime.now()} Random Forest Classifier :')
fold_n = 1
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, clf.predict(X_train))
    acc_test = accuracy_score(y_test, clf.predict(X_test))

    log.info(f'{datetime.now()} FOLD #{fold_n} TRAIN ACC: {acc_train:.2f} / TEST ACC: {acc_test:.2f}')

    fold_n += 1


clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
clf.fit(X, y)

joblib.dump(clf, model_path / 'model_rdf.plk')
log.info(f'{datetime.now()} model for {pos_name} generated successfully')

log.info(f'{datetime.now()} SVM Classifier :')
fold_n = 1
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf = svm.SVC(kernel = 'rbf')
    clf.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, clf.predict(X_train))
    acc_test = accuracy_score(y_test, clf.predict(X_test))

    log.info(f'FOLD #{fold_n} TRAIN ACC: {acc_train} / TEST ACC: {acc_test}')

    fold_n += 1


clf = svm.SVC(kernel = 'rbf', probability=True)
clf.fit(X, y)

joblib.dump(clf, model_path /  'model_svm.plk')
log.info(f'{datetime.now()} model for {pos_name} generated successfully')