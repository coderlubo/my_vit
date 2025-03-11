from utils.myLog import set_logger
logger = None

def set_global_logger():
    global logger
    logger = set_logger()

def get_global_logger():
    global logger
    return logger

