import logging

def get_logger(logger_name: str) -> logging.Logger:
    log = logging.getLogger(logger_name)
    if len(log.handlers) == 0:
        formatter = logging.Formatter(
            "[%(levelname)s][%(asctime)s - Mod: %(module)s - Func: %(funcName)s - Line: %(lineno)s]: %(message)s", "%Y-%m-%d %H:%M:%S")
        s_handler = logging.StreamHandler()
        s_handler.setFormatter(formatter)
        log.addHandler(s_handler)
        f_handler = logging.FileHandler('output.log')
        f_handler.setFormatter(formatter)
        log.addHandler(f_handler)
        log.setLevel(logging.DEBUG)
    return log
