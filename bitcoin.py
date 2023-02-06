#!/usr/bin/env python
# coding: utf-8

# ## Colab 설정 및 필요 라이브러리 설치

# ## 라이브러리 로드

# In[4]:


import warnings
warnings.filterwarnings(action='ignore')

import datetime
from tqdm import tqdm

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_absolute_error

from mlxtend.preprocessing import minmax_scaling

import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

plt.rc('font', family='NanumBarunGothic') 

import lightgbm as lgb
from xgboost import XGBRegressor


# In[5]:


pd.set_option('display.max_rows', 70)
pd.set_option('display.max_columns', None)
path = "/gdrive/My Drive/Brunch/"


# ### 데이터 로드

# In[6]:


import pyupbit

coins = pyupbit.get_tickers(fiat="KRW")
len(coins)


# In[7]:


## 코인 거래 데이터 다운로드를 위한 함수 설정

def get_df(coin):
#     df = pyupbit.get_ohlcv(coin, count=365) # 1년치 데이터 활용
#     df_4h = pyupbit.get_ohlcv(coin, interval = 'minute240', count=365*6) # 240분 단위 데이터 활용

    df = pyupbit.get_ohlcv(coin, interval = 'minute5', count=10*24*12) # 5분 단위 데이터
    df_4h = pyupbit.get_ohlcv(coin, interval = 'minute5', count=10*24*12) # 5분 단위 데이터

    ## 변수 생성
    ## RSI 지표 생성
    df_4h['U'] = df_4h['close'] - df_4h['open']
    df_4h['D'] = df_4h['close'] - df_4h['open']
    df_4h['U'] = np.where(df_4h['U'] > 0, df_4h['U'], 0)
    df_4h['D'] = np.where(df_4h['D'] < 0, df_4h['D'], 0)

    df_4h.index = df_4h.index.strftime("%Y%m%d%H%M")
    df.index = df.index.strftime("%Y%m%d%H%M")

    df_A = df_4h.reset_index().groupby('index').mean()[['U','D']].reset_index()
    df_A.columns = ['index','AU','AD']

    df = pd.merge(df.reset_index(), df_A, on='index', how='left')
    df['price'] = df['close']*100 / df['open']

    df['RSI'] = df['AU'] / (df['AU'] + df['AD'])
    df.drop(['open','high','low','volume','AU','AD'], axis=1, inplace=True)
    
    for i in df.index:
        cur_rsi = df.loc[i, 'RSI']
        try:
            int(cur_rsi)
        except:
            df.loc[i, 'RSI'] = 0.0

    df['RSI'] = df['RSI'].apply(lambda x: df[np.isnan(df['RSI'])].max()['RSI'] if np.isnan(x) else x)
    df['RSI'].describe()

    ## 각 변수의 과거 시점 값을 변수로 생성
    df = pd.concat([df[['index','close','price']],minmax_scaling(df, columns=['value','RSI'])], axis=1)
    lag = np.arange(1,30).tolist()

    close_cols = ['close_' + str(a) for a in lag]
    value_cols = ['value_' + str(a) for a in lag]
    RSI_cols = ['RSI_' + str(a) for a in lag]

    data = pd.DataFrame()
    data['date'] = df['index']
    data['name'] = coin.split('-')[1]

    data['close'] = df['close']
    data['value'] = df['value']
    data['RSI'] = df['RSI']

    for a,b,c,d in zip(lag, close_cols, value_cols, RSI_cols):
        data[b] = df['close'].shift(int(a))
        data[c] = df['value'].shift(int(a))
        data[d] = df['RSI'].shift(int(a))

    time_now = datetime.datetime.now() + datetime.timedelta(days=-1)
    time_now = time_now.strftime("%Y%m%d")

    ## 5일 뒤 종가를 예측 변수로 설정
    ## 1시간 뒤의 종가를 예측 변수로 변경함
    data['target'] = df['close'].shift(-12) # 5분 * 12 = 60분
    data['target'] = data['target']*100 / df['close']

    data['target'] = np.where(data['date'] <= time_now, data['target'], 100)

    data = data.dropna(axis=0)
    data.reset_index(drop = True, inplace=True)

    return data


# In[8]:


## 설정 함수를 이용한 실제 데이터 셋 생성
df = pd.DataFrame()

for coin in tqdm(coins):
    tmp = get_df(coin)
    df = pd.concat([df,tmp])
    break


df.reset_index(drop = True, inplace=True)


# In[9]:


df.iloc[670:]


# ### 머신러닝 모델링

# In[10]:


## 학습, 예측 셋 구분 (최근 5일치 기준)
time_now = datetime.datetime.now() + datetime.timedelta(days=-1)
time_now = time_now.strftime("%Y%m%d")

time_now


# In[11]:


## 학습, 예측 셋 구분 (최근 5일치 기준)
time_now = datetime.datetime.now() + datetime.timedelta(days=-1)
time_now = time_now.strftime("%Y%m%d")

Train = df[df['date'] <= time_now]
Test = df[df['date'] > time_now]

Train.reset_index(drop = True, inplace=True)
Test.reset_index(drop = True, inplace=True)

## 활용 변수 셋팅
cols = df.columns.tolist()
cols.remove('target')
cols.remove('name')
cols.remove('date')

## 학습용, 검증용 데이터셋 분리
X_train, X_valid, y_train, y_valid = train_test_split(Train[cols], Train['target'], train_size=0.8,random_state=42)


# In[12]:


Test


# In[13]:


## XGB를 이용한 머신러닝
xgb = XGBRegressor()
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_valid)
print(mean_absolute_error(y_valid, y_pred))


# In[14]:


## 예측값과 실제값의 차이 확인
tmp = pd.DataFrame()
tmp['pred'] = y_pred
tmp['true'] = y_valid

## 예측이 잘 된 경우와 되지 않은 경우 확인
tmp['diff'] = abs(tmp['pred'] - tmp['true'])
#display(tmp.sort_values('diff', ascending = True).head(10), tmp.sort_values('diff', ascending = False).head(10))


# In[15]:


## 변수 중요도 확인
feature_important = xgb.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

tmp = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
#tmp.plot(kind='barh', figsize=(14,12))


# In[16]:


## LGBM을 활용한 머신러닝 적용

train, valid = train_test_split(Train, train_size=0.8,random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(train[cols], train['target'], train_size=0.8,random_state=42)
                                                      
## LGB
train_ds = lgb.Dataset(X_train, label=y_train)
val_ds = lgb.Dataset(X_valid, label=y_valid)

## 파라미터 셋팅
params = {
            'learning_rate' : 0.05,
            'boosting_type': 'gbdt',
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'metric': 'mae',
            'sub_row' : 0.75,
            'lambda_l2' : 0.1
        }

## 모델 학습
model_t = lgb.train(params,
                train_ds,
                2000,
                val_ds,
                verbose_eval = 100,
                early_stopping_rounds = 100
                )

y_pred = model_t.predict(valid[cols])
#print(mean_absolute_error(valid['target'], y_pred))


# In[17]:


## 성능 확인
#print(mean_absolute_error(valid['target'].values, y_pred))


# In[18]:


## LGBM의 K-Fold 적용
train_set, valid_set = train_test_split(Train,train_size=0.85,random_state=42)
X_train = train_set[cols]
y_train = train_set['target']
X_valid = valid_set[cols]
y_valid = valid_set['target']

## 10 Fold Cross Validation
y_cat = pd.cut(y_train, 10, labels=range(10))
skf = KFold(5)

## 파라미터 셋팅
params = {
            'learning_rate' : 0.05,
            'boosting_type': 'gbdt',
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'metric': 'mae',
            'sub_row' : 0.75,
            'lambda_l2' : 0.1
        }

preds = []
preds_test = []

i = 1
## 실제 학습 진행
for tr_id, val_id in skf.split(X_train, y_cat) : 
    X_tr = X_train[cols].iloc[tr_id]
    y_tr = y_train.iloc[tr_id]

    train_x, valid_x, train_y, valid_y = train_test_split(X_tr, y_tr, train_size=0.8,random_state=42)
    train_ds = lgb.Dataset(train_x, label=train_y)
    val_ds = lgb.Dataset(valid_x, label=valid_y)

    print('{}번째 학습'.format(i))
    model = lgb.train(params,
                  train_ds,
                  2000,
                  val_ds,
                  verbose_eval = 100,
                  early_stopping_rounds = 100
                 ) 
    pred = model.predict(X_valid[cols])
    preds.append(pred)
    i += 1

    pred_test = model.predict(Test[cols])
    preds_test.append(pred_test)

## 예측값 생성 및 점수 확인
model_pred = np.mean(preds, axis = 0)
#print(mean_absolute_error(y_valid, model_pred))


# ### 결과 확인

# In[19]:


## 최종 예측된 데이터셋 생성
final_pred = np.mean(preds_test, axis = 0)

final = pd.DataFrame()
final['coin'] = Test['name']
final['date'] = Test['date']
final['preds'] = final_pred


# In[25]:


Test


# In[22]:


final.iloc[-12:]


# ## table에 추가

# In[26]:


import pymysql


# In[ ]:





# In[41]:


## groups table에 추가
db = pymysql.connect(
                    host= "us-cdbr-east-06.cleardb.net",
                    user= "bbcbf72b01656b",
                    passwd= "f69cff64",
                    db= "heroku_c3b4550615fb803",
                    charset='utf8'
                    )

cursor = db.cursor()

for idx, i in enumerate(range(-12, -2)):
    date = final.iloc[i]['date']
    preds = final.iloc[i]['preds']
    
    pred_for_next = int(Test[Test['date'] == date]['close'] * preds) // 100
    print(idx, pred_for_next)

    # SQL 문 만들기
    sql = f'''UPDATE groups SET ai_price_{idx+1} = {pred_for_next} WHERE group_type = 1'''
    
    # SQL 실행하기
    cursor.execute(sql)
    results = cursor.fetchall()
    
    # 실행 mysql 서버에 확정 반영하기
    db.commit()

# DB 연결 닫기
db.close()


# In[73]:


db_date_list = []


# In[74]:


## bitcoin_price table에 추가

db = pymysql.connect(
  host= "us-cdbr-east-06.cleardb.net",
  user= "bbcbf72b01656b",
  password= "f69cff64",
  database= "heroku_c3b4550615fb803"
)

cursor = db.cursor()

for idx, i in enumerate(range(-12, -2)):
    date = final.iloc[i]['date']
    preds = final.iloc[i]['preds']
    
    pred_for_next = int(Test[Test['date'] == date]['close'] * preds) // 100
    pred_date = str(int(date)+100)
    db_date_list.append(pred_date)
    print(pred_date)
    print(idx, pred_for_next)
    print()
    
    
    # SQL 문 만들기
    sql = f'''UPDATE bitcoin_price SET ai_recommendation = {pred_for_next}, timestamp = {pred_date} WHERE test_num = {idx+1}'''
    
    # SQL 실행하기
    cursor.execute(sql)
    results = cursor.fetchall()
    
    # 실행 mysql 서버에 확정 반영하기
    db.commit()

# DB 연결 닫기
db.close()


# ## Real Price 계산 및 db 추가

# In[75]:


db_date_list


# In[71]:


from datetime import datetime

now = datetime.now()

year = str(now.year)

if len(str(now.month)) == 1:
    month = str(0) + str(now.month)
else:
    month = str(now.month)

if len(str(now.day)) == 1:
    day = str(0) + str(now.day)
else:
    day = str(now.day)
    
if len(str(now.hour)) == 1:
    hour = str(0) + str(now.hour)
else:
    hour = str(now.hour)
    
if len(str(now.minute)) == 1:
    minute = str(0) + str(now.minute)
else:
    minute = str(now.minute)
    
current_date = year + month + day + hour + minute


# In[81]:


import requests
response = requests.get('https://api.upbit.com/v1/candles/minutes/5?market=KRW-BTC&count=100')
json_response = response.json()

db = pymysql.connect(
  host= "us-cdbr-east-06.cleardb.net",
  user= "bbcbf72b01656b",
  password= "f69cff64",
  database= "heroku_c3b4550615fb803"
)

cursor = db.cursor()

for element in json_response:
    date = element['candle_date_time_kst']
    trade_price = element['trade_price']

    str_date = date[0:4] + date[5:7] + date[8:10] + date[11:13] + date[14:16]
    
    if str_date in db_date_list:
        print(str_date)
        print(trade_price)
        print()
        
        db_date_list.remove(str_date)
        
        # SQL 문 만들기
        sql = f'''UPDATE bitcoin_price SET price = {trade_price} WHERE timestamp = {str_date}'''

        # SQL 실행하기
        cursor.execute(sql)
        results = cursor.fetchall()

        # 실행 mysql 서버에 확정 반영하기
        db.commit()

# DB 연결 닫기
db.close()


# ### 아직 추가되지 않은 price에 대해, while문으로 반복

# In[85]:


import time


# In[90]:


t0 = time.time()

while db_date_list:
    # 5분마다 실행
    
    if time.time() - t0 < 300:
        continue
        
    print("Processing...")
    
    response = requests.get('https://api.upbit.com/v1/candles/minutes/5?market=KRW-BTC&count=5')
    json_response = response.json()

    db = pymysql.connect(
      host= "us-cdbr-east-06.cleardb.net",
      user= "bbcbf72b01656b",
      password= "f69cff64",
      database= "heroku_c3b4550615fb803"
    )

    cursor = db.cursor()
    
    for element in json_response:
        date = element['candle_date_time_kst']
        trade_price = element['trade_price']

        str_date = date[0:4] + date[5:7] + date[8:10] + date[11:13] + date[14:16]

        if str_date in db_date_list:
            print(str_date)
            print(trade_price)
            print()

            db_date_list.remove(str_date)

            # SQL 문 만들기
            sql = f'''UPDATE bitcoin_price SET price = {trade_price} WHERE timestamp = {str_date}'''

            # SQL 실행하기
            cursor.execute(sql)
            results = cursor.fetchall()

            # 실행 mysql 서버에 확정 반영하기
            db.commit()
    
    t0 = time.time()
    
    # DB 연결 닫기
    db.close()


# In[ ]:




