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

SEQ_LEN = 150
FUTURE_PERIOD_PREDICT = 20
RATIO_TO_PREDICT = "eurusd"
EPOCHS = 4
BATCH_SIZE = 64
PERIOD = 14
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED{int(time.time())}"
atr_name = "ATR {period}".format(period=PERIOD)
rsi_name = "RSI {period}".format(period = PERIOD)


def preprocess_df(df):
    df = df.drop("future",1)
    for col in df.columns:
        if (col!="target" and col!=atr_name and col!=rsi_name):
            df[col]=df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]]) # Ovo ubacuje sve iz jedne vrste osim target-a
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days),i[-1]]) # Prvo je feature a drugo je label

    # random.shuffle(sequential_data)
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq,target])
        elif target==1:
            buys.append([seq,target])
    # random.shuffle(buys)
    # random.shuffle(sells)

    lower = min(len(buys),len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    # Ovo se radi da bi bilo izbalansirano, jednak broj buys i sells, pa se nalazi manji i onda kaze je buys = buys do lower
    # sequential_data = buys + sells
    # random.shuffle(sequential_data)

    X = []
    y = []

    for seq,targets in sequential_data:
        X.append(seq)
        y.append(targets)

    return np.array(X),y


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------


pickle_in = open("X.pickle","rb")
df = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
validation_df = pickle.load(pickle_in)


train_x,train_y = preprocess_df(df)
validation_x,validation_y = preprocess_df(validation_df)

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

# print(f"train data: {len(train_x)} validation: {len(validation_x)}")
# print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
# print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(LSTM(128,input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128,input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.001,decay=1e-6) # lr je learning rate

model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

history = model.fit(
    train_x,train_y,
    batch_size=BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = (validation_x,validation_y),
    callbacks = [tensorboard,checkpoint])



