import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler

# データの読み込み
data = pd.read_csv('./dataset/MRTSSM45321USN.csv')

# 日付をインデックスに設定
data['observation_date'] = pd.to_datetime(data['observation_date'])
data.set_index('observation_date', inplace=True)

# データのスケーリング
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# データの分割
train_data, test_data = train_test_split(scaled_data, test_size=0.2, shuffle=False)

# RNN用のデータセット作成
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 12  # 過去12ヶ月のデータを使用
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# データの形状を変更
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# RNNモデルの構築
model = Sequential()
model.add(SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(SimpleRNN(50))
model.add(Dense(1))

# モデルのコンパイル
model.compile(optimizer='adam', loss='mean_squared_error')

# モデルの訓練
model.fit(X_train, y_train, epochs=100, batch_size=32)

# 精度の検証
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)  # スケーリングを元に戻す
actual = scaler.inverse_transform(test_data[time_step + 1:])

# 結果の表示
import matplotlib.pyplot as plt

plt.plot(actual, label='Actual')
plt.plot(predicted, label='Predicted')
plt.title('RNN Model Prediction')
plt.xlabel('Time')
plt.ylabel('MRTSSM45321USN')
plt.legend()
plt.show()
