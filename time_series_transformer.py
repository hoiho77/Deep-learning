# 필요한 라이브러리 임포트
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader, TensorDataset
import warnings
from utils.data_preprocessing import *
from utils.utils import *
from utils.model_train import *
from utils.model_test import *
from utils.model_predict import *
import argparse

# 메인 함수
def main(args):
    args.logger, args.log_path = log(args)

    # 프로그램 시작 시간 기록
    args.logger.info("==========PROGRAM START==========")

    data= pd.read_csv(f'{args.data_path}', encoding='cp949', index_col=0)

    # 랜덤 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.logger.info(f'device: {args.device}')
    args.input_dim = data.shape[1] - 1 if args.multi else 1

    if args.mode == 'train' or 'test':
        # data전처리적용
        train_data = preprocessing_data(args, data, args.logger)
        train_data.get_train_data()

    # 모델 훈련 실행
        if args.model_nm == 'rnn':
            for model_nm in ['lstm', 'gru', 'rnn']:  # 'gru', 'rnn' 'lstm',
                model, rnn_history, args.best_model = train_rnn_model(args, train_data, model_nm, args.logger)
                rnn_result(args, data, train_data, model, model_nm, args.logger)

        elif args.model_nm == 'tf':
            model, transformer_history, args.best_model = train_tstrans(args, train_data, args.logger)
            tf_result(args, data, train_data, model, args.logger)

    elif args.mode == 'predict': 
        predict_data = preprocessing_data(args, data, args.logger)
        predict_data.get_test_data()
        
        if args.model_nm != 'tf':
            rnn_model_predict(args, predict_data, args.logger)
            
        elif args.model_nm == 'tf':
            tf_model_predict(args, predict_data, args.logger)
      

        # 프로그램 종료 시간 기록
    args.logger.info("==========PROGRAM FINISH==========")


if __name__ == '__main__':
    import os
    print(os.getcwd())

    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='time-series deep-learning training')

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--model_nm', type=str, default='tf')
    parser.add_argument('--input_window', type=int, default=100)
    parser.add_argument('--output_window', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--multi', type=bool, default=False)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='./data.csv')
    parser.add_argument('--seed', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default=None)


    # args 에 위의 내용 저장
    args = parser.parse_args()
    args.criterion = nn.MSELoss()

    args.file_name = f'{args.model_nm}_{args.multi}_{args.input_window}_{args.output_window}_{args.num_epochs}_{args.lr}'


    # main 파일 돌리기
    main(args)