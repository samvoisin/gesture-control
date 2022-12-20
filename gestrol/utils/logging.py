# standard libraries
import logging
import sys


def configure_logging():
    """
    Set basic logging configuration. Info level. Sends to stdout.
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s] (%(filename)s:%(lineno)d) %(levelname)s : %(message)s",
    )
