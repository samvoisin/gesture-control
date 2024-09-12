import logging
from typing import Optional


def configure_logger(logfile: Optional[str] = None) -> logging.Logger:
    """
    Configure logger.

    Args:
        logfile (Optional[str], optional): Path to log file. Defaults to None.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger("GestuReMote")

    # check if logger already has handlers (to avoid duplicate logs)
    if not logger.handlers:
        print(logfile)
        # create a console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if logfile:
            # create a file handler
            file_handler = logging.FileHandler(logfile)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
