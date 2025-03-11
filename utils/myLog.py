import logging
import os
import time

from utils.settings import LOG_DIR, LOG_PATH

def set_logger():

    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    # 创建日志输出目录
    if os.path.exists(LOG_DIR) is False:
        os.makedirs(LOG_DIR)
    
    # 建立一个filehandler来把日志记录在文件里，级别为debug以上
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.DEBUG)

    # 建立一个streamhandler来把日志打在CMD窗口上，级别为error以上
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 设置日志格式
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(lineno)s %(message)s", )
    formatter = logging.Formatter("%(message)s")
    
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    #将相应的handler添加在logger对象中
    logger.addHandler(fh)

    path = os.path.basename(os.path.dirname(os.path.dirname(__file__))) + "/main.py"
    logger.info(path)
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime()))
    
    logger.addHandler(ch)

    return logger