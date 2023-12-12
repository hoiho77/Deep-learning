import torch
import torch.optim as optim
from utils.build_model import *
import os
import numpy as np
import pandas as pd


def rnn_model_predict(args, data, logger):
    m = args.multi
    window = args.input_window
    output_window = args.output_window
    input_dim = args.input_dim
    num_layers = args.num_layers
    device = args.device
    output_dim = window
    model_nm = args.model_nm
    num_rnn_layers = args.num_layers
    hidden_dim = args.hidden_dim
    criterion = args.criterion


    if model_nm == 'rnn':
      rnn_model = RNNModel(input_dim, hidden_dim, num_rnn_layers, output_window, m).to(device)

    elif model_nm == 'lstm':
      rnn_model = LSTMModel(input_dim, hidden_dim, num_rnn_layers, output_window, m).to(device)

    elif model_nm == 'gru':
      rnn_model = GRUModel(input_dim, hidden_dim, num_rnn_layers, output_window, m).to(device)

    rnn_model.load_state_dict(torch.load(f"{args.model_path}"))
    logger.info(f'{model_nm} 모델을 불러왔습니다!')

    src_seq = data.sequences.to(device) if args.multi else data.sequences.unsqueeze(-1).to(device)

    tmp = pd.DataFrame()
    for i in range(0, len(src_seq), 100):
        try:
            batch_X = src_seq[i:i+100]
            prediction = rnn_model(batch_X)

        except:
            batch_X = src_seq[i:]
            prediction = rnn_model(batch_X)

        tmp = pd.concat([tmp, pd.DataFrame(prediction.squeeze(-1).cpu().detach().numpy())])
    
    tmp.index = data.index_info[:len(tmp)]
    output_df = tmp.copy()
    output_df.to_csv(f'{args.log_path}/prediction.csv')

    if output_window == 1:
        plt.plot(output_df.index, output_df.iloc[:, -output_window].tolist())

    else:
        fig, ax = plt.subplots(output_window, 1)
        for i in range(output_window):
            ax[i].plot(output_df.index, output_df.iloc[:, -output_window + i].tolist())

    plt.savefig(f'{args.log_path}/prediction_plot.png')


def tf_model_predict(args, data, logger):
    m = args.multi
    window = args.input_window
    output_window = args.output_window
    input_dim = args.input_dim
    d_model = args.d_model
    num_heads = args.num_heads
    batch_size = args.batch_size
    num_layers = args.num_layers
    device = args.device
    model_nm = args.model_nm
    max_len = window
    output_dim = window

    transformer_model = TsTrans(input_dim, max_len, output_dim, d_model, num_heads, num_layers, batch_size).to(device)
    transformer_model.load_state_dict(torch.load(f"{args.model_path}"))
    logger.info(f'{model_nm} 모델을 불러왔습니다!')

    src_seq = data.sequences.to(device) if args.multi else data.sequences.unsqueeze(2).to(device)

    tmp = pd.DataFrame()
    for i in range(0, len(src_seq), 100):
        try:
            batch_X = src_seq[i:i+100]
            prediction = transformer_model(batch_X)

        except:
            batch_X = src_seq[i:]
            prediction = transformer_model(batch_X)

        tmp = pd.concat([tmp, pd.DataFrame(prediction.squeeze(-1)[:, -output_window:].cpu().detach().numpy())])
    
    tmp.index = data.index_info[:len(tmp)]
    output_df = tmp.copy()
    output_df.to_csv(f'{args.log_path}/prediction.csv')

    if output_window == 1:
        plt.plot(output_df.index, output_df.iloc[:, -output_window].tolist())

    else:
        fig, ax = plt.subplots(output_window, 1)
        for i in range(output_window):
            ax[i].plot(output_df.index, output_df.iloc[:, -output_window + i].tolist())

    plt.savefig(f'{args.log_path}/prediction_plot.png')
        #plt.suptitle('Prediction Graph')