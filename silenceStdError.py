"""
Context manager for silencing stderr (e.g. TensorFlow messages)
"""

import os
import sys

class nullcontext:
    """
    Context manager that does nothing, for use in conditional context management.
    Usage:
    with nullcontext():
        do something
    """
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc, tb):
        pass

class SilenceStdErr:
    """
    Context manager for silencing stderr (e.g. TensorFlow messages)
    :param: None
    :return: None
    :raises: None
    .. note::
        works by redirecting the file descriptor for stderr to os.devnull
    .. warning::
        for demonstration purposes only, not recommended for debugging during development or production
        does not work in Jupyter notebooks
        does not work on Windows
    .. seealso::
        https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    Usage:
    with silence_stderr():
        do something that prints to stderr
    """
    def __enter__(self):
        self._stderr_fd = sys.stderr.fileno()
        self._saved = os.dup(self._stderr_fd)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, self._stderr_fd)
        os.close(self._devnull)
    def __exit__(self, exc_type, exc, tb):
        os.dup2(self._saved, self._stderr_fd)
        os.close(self._saved)
