import os
import numpy as np
import torch
import torch.optim as optim
from utils.build_model import *
from torchsummary import summary
import torch.optim as optim

def train_rnn_model(args, rnn_data, model_nm, logger):
    num_rnn_layers = args.num_layers
    output_window = args.output_window
    hidden_dim = args.hidden_dim
    input_dim = args.input_dim
    lr = args.lr
    criterion = args.criterion
    num_epochs = args.num_epochs
    m = args.multi
    batch_size = args.batch_size
    device = args.device

    if model_nm == 'rnn':
      rnn_model = RNNModel(input_dim, hidden_dim, num_rnn_layers, output_window, m).to(device)

    elif model_nm == 'lstm':
      rnn_model = LSTMModel(input_dim, hidden_dim, num_rnn_layers, output_window, m).to(device)

    elif model_nm == 'gru':
      rnn_model = GRUModel(input_dim, hidden_dim, num_rnn_layers, output_window, m).to(device)

    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=lr)
    rnn_history = {'loss': [], 'val_loss': []}
    logger.info(f"{model_nm} : {sum(p.numel() for p in rnn_model.parameters() if p.requires_grad)}")

    if args.mode=='test':
      return rnn_model, None, None

    model_path = f"{args.log_path}/model"
    
    logger.info(rnn_model)

    if not os.path.exists(model_path):
        os.makedirs(f"{model_path}")

    min_val_loss = np.inf

    for epoch in range(num_epochs):
      running_loss =0
      rnn_model.train()

      for i in range(0, len(rnn_data.train_sequences), batch_size):
          batch_X = rnn_data.train_sequences[i:i + batch_size].to(device)
          batch_y = rnn_data.train_targets[i:i + batch_size].to(device)

          rnn_optimizer.zero_grad()
          if m:
              rnn_output = rnn_model(batch_X)
          else:
              rnn_output = rnn_model(batch_X.unsqueeze(-1))

          rnn_loss = criterion(rnn_output, batch_y.to(device))
          running_loss += rnn_loss.item()
          rnn_loss.backward()
          rnn_optimizer.step()
      length = len(list(range(0, len(rnn_data.train_sequences), batch_size)))
      print('train')
      # Validation 손실 계산 및 출력

      t = torch.zeros([1, output_window])
      with torch.no_grad():
          rnn_model.eval()

          for idx in range(0, len(rnn_data.valid_sequences), 100):
              batch_X = rnn_data.valid_sequences[idx:idx + 100]
              batch_X = batch_X if m else batch_X.unsqueeze(-1)
              rnn_val_loss = rnn_model(batch_X.to(device))
              t = torch.concat((t.to(device), rnn_val_loss), dim=0)
              
          rnn_val_loss = criterion(t[1:,:], rnn_data.valid_targets.to(device))

          if min_val_loss > rnn_val_loss:
                best_model = f"best_model_{model_nm}_{epoch+1}.pth"
                torch.save(rnn_model.state_dict(), f"{model_path}/{best_model}")
                min_val_loss = rnn_val_loss

          rnn_history['loss'].append(running_loss/length)
          rnn_history['val_loss'].append(rnn_val_loss.item())
          logger.info(f'Epoch [{epoch+1}/{num_epochs}], {model_nm} Train Loss:{running_loss/length:.4f}, {model_nm} Val Loss: {rnn_val_loss:.4f}')

    plt.plot([i for i in range(num_epochs)], rnn_history['loss'], label='loss')
    plt.plot([i for i in range(num_epochs)], rnn_history['val_loss'], label='val_loss')
    plt.savefig(f'{args.log_path}/loss_{args.file_name}.png')
    plt.clf()
    logger.info(f"{model_nm}_베스트 모델: {best_model}")
    return rnn_model, rnn_history, best_model


def train_tstrans(args, tf_data, logger):
    m = args.multi
    window = args.input_window
    output_window = args.output_window
    input_dim = args.input_dim
    d_model = args.d_model
    num_heads = args.num_heads
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    criterion = args.criterion
    learning_rate = args.lr
    num_layers = args.num_layers
    device = args.device
    model_nm = args.model_nm
    max_len = window
    output_dim = window

    logger.info(f"input_dim={input_dim}, max_len={max_len}, output_dim={output_dim}, d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}")
    logger.info(f"batch_size={batch_size}, learning_rate={learning_rate}")

    transformer_model = TsTrans(input_dim, max_len, output_dim, d_model, num_heads, num_layers).to(device)
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate)
    #scheduler = optim.lr_scheduler.StepLR(transformer_optimizer, step_size=5, gamma=0.7)

    logger.info(f"transformer_model : {sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)}")

    # 모델로 추론만 할 경우는 model_path 및 mode='test'지정
    if args.mode=='test':
      return transformer_model, None, None

    # 학습을 추가로 시키고 싶은 모델이 있을 경우는 model_path 및 mode='train'지정
    if args.model_path!=None:
      transformer_model.load_state_dict(torch.load(f"{args.model_path}"))

    transformer_history = {'loss': [], 'val_loss': []}

    model_path = f"{args.log_path}/model"
    if not os.path.exists(model_path):
        os.makedirs(f"{model_path}")
        
    best_model =''
    min_val_loss = np.inf
    for epoch in range(num_epochs):
        running_loss = 0
        transformer_model.train()

        for i in range(0, len(tf_data.train_sequences), batch_size):
            batch_X = tf_data.train_sequences[i:i + batch_size].to(device)
            batch_y = tf_data.train_targets[i:i + batch_size].to(device)

            transformer_optimizer.zero_grad()

            src_seq = batch_X[:, :] if m else batch_X.unsqueeze(2)

            transformer_output = transformer_model(src_seq)
            transformer_loss = criterion(transformer_output.squeeze(-1)[:, -output_window:], batch_y[:, -output_window:])

            running_loss += transformer_loss.item()
            transformer_loss.backward()
            transformer_optimizer.step()

        length = len(list(range(0, len(tf_data.train_sequences), batch_size)))

        # Validation 손실 계산 및 출력
        t = torch.zeros([1, output_window, 1])
        with torch.no_grad():
            transformer_model.eval()
            for idx in range(0, len(tf_data.valid_sequences), 100):
              batch_X = tf_data.valid_sequences[idx:idx + 100].to(device)
              batch_X = batch_X if m else batch_X.unsqueeze(2)
              transformer_val_output = transformer_model(batch_X)
              t = torch.concat((t.to(device),transformer_val_output[:, -output_window:]), dim=0)
            
            transformer_val_loss = criterion(t[1:,:,:].squeeze(-1), tf_data.valid_targets[:, -output_window:].to(device))

            if min_val_loss > transformer_val_loss:
                best_model = f"best_model_{model_nm}_{epoch+1}.pth"
                torch.save(transformer_model.state_dict(), f"{model_path}/{best_model}")
                min_val_loss = transformer_val_loss

        transformer_history['loss'].append(running_loss /length)
        transformer_history['val_loss'].append(transformer_val_loss.item())
        logger.info(f'{model_nm} : Epoch [{epoch + 1}/{num_epochs}], Transformer Train Loss: {running_loss / length:.4f}, Transformer Val Loss: {transformer_val_loss:.4f}')
        #scheduler.step()

    plt.plot([i for i in range(num_epochs)], transformer_history['loss'], label='loss')
    plt.plot([i for i in range(num_epochs)], transformer_history['val_loss'], label='val_loss')
    plt.title('Train/Valid loss')
    plt.savefig(f'{args.log_path}/loss_plot.png')
    plt.clf()

    logger.info(f"{model_nm}_베스트 모델: {best_model}")

    return transformer_model, transformer_history, best_model

