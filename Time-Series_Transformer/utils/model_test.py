import matplotlib.pyplot as plt
import torch 

# 시각화 설정
#plt.rcParams['font.family'] = 'Malgun Gothic' #NanumBarunGothic
plt.rcParams['axes.unicode_minus'] = False

def rnn_result(args, data, rnn_data, model, model_nm, logger):
    torch.cuda.empty_cache()
    torch.cuda.max_memory_allocated()
    
    device = args.device
    if args.mode =='train':
      model.load_state_dict(torch.load(f"{args.log_path}/model/{args.best_model}"))
    elif args.mode =='test':
      model.load_state_dict(torch.load(f"{args.model_path}"))
    
    logger.info('best_model을 불러왔습니다.')
    
    model.eval()
    model.cpu()

    t = torch.zeros([1, args.output_window])
    for idx in range(0, len(rnn_data.test_sequences), 2):
      batch_X = rnn_data.test_sequences[idx:idx + 2]#.to(device)
      batch_X = batch_X if args.multi else batch_X.unsqueeze(-1)
      prediction = model(batch_X)
      t = torch.concat((t.cpu(), prediction[:, -args.output_window:].cpu()), dim=0)

    logger.info("예측 완료")
    prediction = t[1:,:]
    logger.info(f'전체 Test Loss: {args.criterion(prediction, rnn_data.test_targets)}')

    if args.output_window == 1:
        fig, ax = plt.subplots(1, 1)
        axes = [ax] 
        
    else:
        fig, axes = plt.subplots(args.output_window, 1)

    for i, ax in enumerate(axes):
        ax.plot(data.index[-len(rnn_data.test_targets):], rnn_data.test_targets[:, i])
        ax.plot(data.index[-len(rnn_data.test_targets):],  prediction[:, i].tolist())
        logger.info(f'{model_nm} loss for t+{i + 1}시점: {args.criterion(prediction[:, i], rnn_data.test_targets[:, i])}')
    plt.savefig(f'{args.log_path}/{model_nm}_testset_prediction_plot_uni.png')
    plt.clf()

def tf_result(args, data, tf_data, tf_model, logger):
    model_nm = args.model_nm
    device = args.device
    
    torch.cuda.empty_cache()
    torch.cuda.max_memory_allocated()
    
    #import gc
    #gc.collect()
    
    if args.mode =='train':
      tf_model.load_state_dict(torch.load(f"{args.log_path}/model/{args.best_model}"))
    elif args.mode =='test':
      tf_model.load_state_dict(torch.load(f"{args.model_path}"))
    
    logger.info('best_model을 불러왔습니다.')

    tf_model.eval()
    tf_model.cpu()

    output_window = args.output_window
    device = args.device
    criterion = args.criterion

    t = torch.zeros([1, output_window, 1])
    for idx in range(0, len(tf_data.test_sequences), 2):
      batch_X = tf_data.test_sequences[idx:idx + 2]#.to(device)
      batch_X = batch_X if args.multi else batch_X.unsqueeze(2)
      prediction = tf_model(batch_X)
      t = torch.concat((t.cpu(), prediction[:, -output_window:].cpu()), dim=0)
    
    logger.info("예측 완료")
    prediction = t[1:,:,:]
    logger.info(f'전체 Test Loss: {criterion(prediction.squeeze(-1), tf_data.test_targets[:, -output_window:])}')

    if output_window == 1:
        plt.plot(data.index[-len(tf_data.test_targets):], tf_data.test_targets[:, -output_window])
        plt.plot(data.index[-len(tf_data.test_targets):], prediction[:, -output_window].tolist())
        logger.info(f'Transformer loss for testset: {criterion(prediction.squeeze(-1)[:, -1], tf_data.test_targets[:, -1])}')
        plt.title('Prediction Graph.png')

    else:
        fig, ax = plt.subplots(output_window, 1)
        for i in range(output_window):
            ax[i].plot(data.index[-len(tf_data.test_targets):], tf_data.test_targets[:, -output_window + i])
            ax[i].plot(data.index[-len(tf_data.test_targets):], prediction[:, -output_window + i].tolist())
            logger.info(f'Transformer loss for testset t+{i + 1}시점: {criterion(prediction.squeeze(-1)[:, -output_window+i], tf_data.test_targets[:, -output_window+i])}')
        plt.suptitle('Prediction Graph')

    plt.savefig(f'{args.log_path}/{model_nm}_testset_prediction_plot_multi.png')
    plt.clf()

