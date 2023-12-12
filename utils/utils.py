from datetime import datetime
import os
import logging

def log(args):
    now = datetime.now()
    today = now.strftime('%Y_%m_%d')
    hours = now.strftime('%H_%M_%S')
    file_path = f"./record/{args.mode}/{args.model_nm}_{today}/"
    log_path = f"{file_path}/{args.file_name}_{hours}" if args.mode =='train' else f"{file_path}/{args.file_name}_{hours}_{args.mode}"

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger(__name__)

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(f"{log_path}/log_file_{args.file_name}.txt")    # 로그를 기록할 파일 이름 지정

    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    logger.setLevel(level=logging.DEBUG)    # INFO 레벨로 지정하면, INFO 레벨보다 낮은 DEBUG 로그는 무시함(기본: WARNING으로 설정)

    return logger, log_path
