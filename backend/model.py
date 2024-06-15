path='tsm2330_100epoch_0.019.h5'
# !pip install yfinance
# !pip install optuna
import yfinance as yf
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt  # for 畫圖用
import pandas as pd
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import optuna
from sklearn.preprocessing import MinMaxScaler

mode='tranning'

result=[]
# tsm = yf.Ticker('2330.tw')
# tsla = yf.Ticker('TSLA')

yf.download('TSM TSLA',start='2016-01-01',end='2022-01-01')#過去的資料也可以用interval
data_set=yf.download('2330.tw',start='2016-01-01',end='2022-01-01')
#data_set

dataset_train= data_set
training_set = dataset_train.iloc[:, 1:2].values  # 取「Open」欄位值

np.shape(dataset_train)
sc = MinMaxScaler(feature_range = (0, 1))

def processing_data(dataset,mode='train'):
  X = []   #預測點的前 60 天的資料
  y = []   #預測點1151/1258
  global sc,data_set_scaled
  # sc = MinMaxScaler(feature_range = (0, 1))
  print(training_set)
  data_set_scaled = sc.fit_transform(training_set)
  if mode=='train':
    for i in range(60, 1461):  # 1258 是訓練集總數
        #1461
        X.append(data_set_scaled[i-60:i, 0])
        y.append(data_set_scaled[i, 0])

  else:

    dataset_total = pd.concat((dataset_train['Open'], dataset['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values#從train中那60天
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs) # Feature Scaling

    for i in range(60, 458+60):  # timesteps一樣60； 80 = 先前的60天資料
        X.append(inputs[i-60:i, 0])
        y.append(inputs[i-60:i, 0])


    #0 0-60
  X_processed, y_processed = np.array(X), np.array(y)  # 轉成numpy array的格式，以利輸入 RNN
  X_processed = np.reshape(X_processed, (X_processed.shape[0], X_processed.shape[1], 1))

  return X_processed,y_processed

    
if mode=='trainning':#training
    train_x,train_y=processing_data(training_set,'train')

    regressor = Sequential()

    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (train_x.shape[1], 1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# 進行訓練
    regressor.fit(train_x, train_y, epochs = 100, batch_size = 32)

    regressor.save('tsm2330_100epoch.h5')
else:
    regressor=load_model(path)
    
# processing_data(training_set,'train')

def predict(mode='data'):
    
    global dataset_test,real_stock_price,test_x,test_y,predicted_stock_price,score,result
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    if not result:
        dataset_test=yf.download('2330.tw',start='2022-01-02',end='2023-11-23',interval='1d')
        real_stock_price = dataset_test.iloc[:, 1:2].values

        test_x,test_y=processing_data(dataset_test,'test')

        predicted_stock_price = regressor.predict(test_x)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)  # to get the original scale

        np.shape(predicted_stock_price)

        if(mode!='data'):
            plt.plot(real_stock_price, color = 'red', label = 'Real TSM Price')  # 紅線表示真實股價
            plt.plot(predicted_stock_price, color = 'blue', label = 'TSM Price')  # 藍線表示預測股價

            plt.title('TSM Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('TSM Price')
            plt.legend()
            plt.show()

            plt.plot(real_stock_price-predicted_stock_price, color = 'blue', label = 'err')  # 藍線表示預測股價
            plt.show()

        score = regressor.evaluate(test_x,test_y,verbose='2')
        print('score',score)
        print('avr',sum(real_stock_price-predicted_stock_price)/485)
        

        sum(real_stock_price-predicted_stock_price)/485
        # print(real_stock_price.tolist())
        # print(predicted_stock_price.tolist())
        
        #output date
        print(len(predicted_stock_price.tolist()))
        date=dataset_test.index.tolist()

        for col in range(len(date)):
            date[col]= dataset_test.index[col].strftime('%Y/%m/%d')
        # print(date)
        # print(len(date))
        # print(len(predicted_stock_price.tolist()))
        result=predicted_stock_price.tolist(),real_stock_price.tolist(),date

    return result

if __name__=='__main__':
    predict(mode='auto')