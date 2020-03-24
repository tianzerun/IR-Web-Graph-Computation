import logging


def get_logger(logger_name, filename):
    log = logging.getLogger(logger_name)
    f_handler = logging.FileHandler(filename)
    f_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log.addHandler(f_handler)

    return log