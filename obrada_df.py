import pandas as pd
import pickle
import numpy as np
from sklearn import preprocessing
from collections import deque
import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "eurusd"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED{int(time.time())}"


def classify(current,future):
    if float(future)>float(current):
        return 1
    else:
        return 0

def df_ATR(df,period):
    lst = []
    lst.append(0.0001)
    prev_days_ATR = deque(maxlen=period)
    for i in range(1,len(df.index)):
        time_0, open_0, high_0, low_0, close_0, volume_0 = df.iloc[i-1]
        time, open, high, low, close, volume = df.iloc[i]
        tr = max(abs(high-low),abs(high-close_0),abs(close_0-low))
        prev_days_ATR.append(tr)
        ATR = 0
        count = 0
        if(len(prev_days_ATR)==period):
            for ct in range(0,period):
                ATR = ATR + prev_days_ATR[ct]
            ATR = 1/period * ATR
        else:
            for ct in range(0, len(prev_days_ATR)):
                ATR = ATR + prev_days_ATR[ct]
                count = count + 1
            ATR = ATR/count
        lst.append(ATR)
    return lst

def df_RSI(df,period):
    lst = []
    lst.append(50)
    counter = 0
    previous_average_gain=0
    previous_average_loss=0
    first_average_gain = 0
    first_average_loss = 0
    for i in range(1,len(df.index)):
        time, open, high, low, close, volume, atr = df.iloc[i]
        res = close - open
        if res>=0:
            res_gain = res
            res_loss = 0
        else:
            res_gain = 0
            res_loss = abs(res)
        counter+=1
        if(counter<14):
            if(res>=0):
                first_average_gain+=res
            else:
                first_average_loss+=res
        if(counter==14):
            first_average_loss=abs(first_average_loss)/PERIOD
            first_average_gain=abs(first_average_gain)/PERIOD
            previous_average_gain = first_average_gain
            previous_average_loss = first_average_loss
        if(counter>=14):
            average_gain = ((PERIOD-1)*previous_average_gain+res_gain)/PERIOD
            average_loss = ((PERIOD-1)*previous_average_loss+res_loss)/PERIOD
            RS = average_gain/average_loss
            RSI = 100 - (100/(1+RS))
            previous_average_gain = average_gain
            previous_average_loss = average_loss
        if(counter<14):
            RSI = 50
        lst.append(RSI)
    return lst






# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------




PERIOD = 14

df = pd.read_csv("eurusdm30.csv",names=["time","open","high","low","close","volume"])

# for col in df.columns:
#     if col=="volume":
#         df[col]=round(df[col],4)

df['ATR {period}'.format(period = PERIOD)] = pd.DataFrame(df_ATR(df,PERIOD))
df.fillna(method="ffill", inplace=True)
df.dropna(inplace=True)
df['RSI {period}'.format(period=PERIOD)] = pd.DataFrame(df_RSI(df,PERIOD))
df.fillna(method="ffill", inplace=True)
df.dropna(inplace=True)
print(df.tail(50))


df.set_index("time", inplace=True)

# print(df[["close","future"]].head())
df.fillna(method="ffill", inplace=True)
df.dropna(inplace=True)
atr_name = "ATR {period}".format(period=PERIOD)
rsi_name = "RSI {period}".format(period = PERIOD)
df = df[["close","volume",atr_name,rsi_name]]
df['future'] = df["close"].shift(-FUTURE_PERIOD_PREDICT)
df['target'] = list(map(classify,df["close"],df["future"]))
df.dropna(inplace=True)
# map je samo fora da ce funkciju da primeni na sve naredne parametre, iterativno dok sve ne zavrsi, i onda ce sve to
# da pretvori u listu kako bi mogli da ga dodamo na df kao kolonu
# print(df[["close","future","target"]].head(10))

times = sorted(df.index.values)
last_5pct = times[-int(0.05*len(times))]
validation_df = df[(df.index >= last_5pct)]
# df[(uslov)] izvlaci samo podatke koji postuju uslov
df = df[(df.index < last_5pct)]



# Save ujem df u pickle fajl da ne bi morao svaki put da ga pravim
pickle_out = open("X.pickle","wb")
pickle.dump(df,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(validation_df,pickle_out)
pickle_out.close()