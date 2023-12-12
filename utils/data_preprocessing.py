# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# 데이터셋 생성
class preprocessing_data():
    def __init__(self, args, data, logger): # m : 다변량인지 여부
        super(preprocessing_data, self).__init__()
        self.model = args.model_nm
        self.data = data
        self.window = args.input_window
        self.out_window = args.output_window
        self.logger = logger
        self.stride = args.stride
        self.multi = args.multi

    # 시계열 데이터를 모델에 입력할 형태로 변환
    def get_sequence(self):
        X, y = [], []
        if self.multi==True:
          for i in range(0, len(self.data) - self.window - (self.out_window-1), self.stride):

            if self.model == 'tf':
              # 입력차원 = window(t-window 예: t-9 ~ t) / 출력차원 = window(t-window+output_window~t+output_window 예: t-8 ~ t+1) :
              seq = self.data.iloc[i:i+self.window, :-1].values  #ex: t-30~t-1 , 독립변수
              target = self.data.iloc[i+self.out_window:i+self.window+self.out_window, -1].values  # t-27, ..., t t+1 t+2 (out_window=1), 종속변수
            
            else:
              seq = self.data.iloc[i:i+self.window, :-1].values #ex: t-30~t-1 , 독립변수
              target = self.data.iloc[i+self.window:i+self.window+self.out_window, -1].values # t t+1 t+2 (out_window=3), 종속변수


            X.append(seq)
            y.append(target)

        else : # 단변량
          self.data= self.data.iloc[:,-1]
          for i in range(len(self.data) - self.window - (self.out_window-1)):
            if self.model == 'tf':
                seq = self.data.iloc[i:i+self.window].values
                target = self.data.iloc[i+self.out_window:i+self.window+self.out_window].values
            else:
                seq = self.data[i:i+self.window].values
                target = self.data[i+self.window:i+self.window+self.out_window].values

            X.append(seq)
            y.append(target)

        return np.array(X), np.array(y)

    # 데이터를 PyTorch Tensor로 변환
    def transform_to_tensor(self, sequences, targets):
        sequences = torch.FloatTensor(sequences)  # 입력 시퀀스
        targets = torch.FloatTensor(targets)  # 타겟 값
        return sequences, targets

    def get_train_data(self):
        X, y = self.get_sequence()

        if self.multi ==True:
          X_T, X_test, y_T, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
          X_train, X_valid, y_train, y_valid = train_test_split(X_T, y_T, test_size=0.2, shuffle=False)

        else :
          X_train = X[:-1500]
          y_train = y[:-1500]
          X_valid = X[int((len(X)-1500)*0.8):-1500]
          y_valid = y[int((len(X)-1500)*0.8):-1500]
          X_test = X[-1500:]
          y_test = y[-1500:]

          #X_train = X[:int((len(X)*0.8)*0.7)]
          #y_train = y[:int((len(X)*0.8)*0.7)]
          #X_valid = X[int((len(X)*0.8)*0.7):int(len(X)*0.8)]
          #y_valid = y[int((len(X)*0.8)*0.7):int(len(X)*0.8)]
          #X_test = X[int(len(X)*0.8):]
          #y_test = y[int(len(X)*0.8):]

        self.train_sequences, self.train_targets = self.transform_to_tensor(X_train, y_train)
        self.valid_sequences, self.valid_targets = self.transform_to_tensor(X_valid, y_valid)
        self.test_sequences, self.test_targets = self.transform_to_tensor(X_test, y_test)

        self.logger.info(f'훈련 데이터 수: {self.train_sequences.shape}')
        self.logger.info(f'검증 데이터 수: {self.valid_sequences.shape}')
        self.logger.info(f'테스트 데이터 수:{self.test_sequences.shape}')

    def get_test_data(self):
        X, y = self.get_sequence()
        self.sequences, self.targets = self.transform_to_tensor(X, y)
        self.index_info = self.data.index[self.window:]

