import logging


def configure_logger():
    logger = logging.getLogger("GestuReMote")

    # Check if the logger already has handlers (to avoid duplicate logs)
    if not logger.handlers:
        # Create a console handler
        console_handler = logging.StreamHandler()

        # Create a formatter with timestamp
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Attach the formatter to the handler
        console_handler.setFormatter(formatter)

        # Attach the handler to the logger
        logger.addHandler(console_handler)

    return logger
