import logging
from typing import Optional


def setup_logging(
    logfile: Optional[str] = None,
    level: str = logging.INFO,
    include_host: bool = False,
):
    if include_host:
        import socket

        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d,%H:%M:%S',
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S'
        )

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if logfile:
        file_handler = logging.FileHandler(filename=logfile)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
