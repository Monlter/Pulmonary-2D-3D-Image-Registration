import logging
import time
import torch
import numpy as np
from tools.tool_functions import get_logger

def test(filename):

    logger = get_logger(filename,1,filename+"111")
    for i in range(10):
        logger.info(i)
    logging.shutdown()


if __name__ == '__main__':
    test("1.log")
    time.sleep(2)
    test("2.log")
    time.sleep(2)
    test("3.log")


