import logging
import sys

LOG_LEVEL = logging.INFO
FILE_LOG_LEVEL = LOG_LEVEL - 5


def init_root_logger(name):
    # Register a new log level
    logging.addLevelName(FILE_LOG_LEVEL, "INFO_FILE")
    # Create the Logger
    logger = logging.getLogger(name)
    logger.setLevel(FILE_LOG_LEVEL)

    # Create a Formatter for formatting the log messages
    logger_file_formatter = logging.Formatter('%(asctime)s - '
                                              '%(levelname)s - %(message)s',
                                              datefmt='%d-%b-%y %H:%M:%S')

    # Create the Handler for logging data to a file
    logger_file_handler = logging.FileHandler('adder.log', 'w+')
    logger_file_handler.setLevel(FILE_LOG_LEVEL)

    # Add the Formatter to the Handler
    logger_file_handler.setFormatter(logger_file_formatter)

    # Add the Handler to the Logger
    logger.addHandler(logger_file_handler)

    # Repeat the above for a to-screen logger
    logger_stream_handler = logging.StreamHandler()
    logger_stream_handler.setLevel(LOG_LEVEL)
    logger_stream_formatter = logging.Formatter('%(message)s')
    logger_stream_handler.setFormatter(logger_stream_formatter)
    logger.addHandler(logger_stream_handler)
    # Write to the streams so that the root logger is fully configured
    logger.info("ADDER Initialized")

    return logger


def init_logger(name):
    # This is used here and in other functions to abstract the specific
    # logger used
    return logging.getLogger(name)


class LoggedClass(object):
    """This class provides for consistent logger interfaces across
    classes.

    It should be inherited from for all classes that want a logger,
    and the class should call the LoggedClass' init method to initialize
    the logger."""

    def __init__(self, default_indent, name):
        self._default_indent = default_indent
        self._logger = init_logger(name)

    def log(self, level, message, indent=None):
        # Classes call this to write a log
        if indent is None:
            use_indent = self._default_indent
        else:
            use_indent = indent

        if level in ["error", "critical"]:
            use_indent = 0

        low_level = level.lower()
        if low_level in ["info", "warning", "error", "critical", "debug"]:
            func = getattr(self._logger, low_level)
            func(use_indent * " " + message)
            if low_level in ["error", "critical"]:
                sys.exit(1)
        elif low_level in ["info_file"]:
            self._logger.log(FILE_LOG_LEVEL, use_indent * " " + message)
        else:
            raise ValueError("Invalid level: {}".format(low_level))

    def update_logs(self, logs):
        for log_tuple in logs:
            l_type, l_msg, l_indent = log_tuple
            self.log(l_type, l_msg, l_indent)
