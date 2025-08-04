import sys

class Logger(object):
    """
    Logger class that can simultaneously write what's printed to log file
    """

    def __init__(self,logname):
        self.terminal = sys.stdout
        self.log = open(logname, "w",buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


def prettify_time(seconds):
    """
    Converts a time in seconds to a more human-readable format, handling hours, minutes, and seconds.

    Parameters:
        seconds (float): The time in seconds.

    Returns:
        str: The time in a prettified format (hours, minutes, seconds, etc.).
    """
    if seconds >= 3600:  # More than 1 hour
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours}h {minutes}m {seconds:.2f}s"
    elif seconds >= 60:  # More than 1 minute
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}m {seconds:.2f}s"
    elif seconds >= 1:  # More than 1 second
        return f"{seconds:.6f} seconds"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} milliseconds"
    elif seconds >= 1e-6:
        return f"{seconds * 1e6:.3f} microseconds"
    else:
        return f"{seconds * 1e9:.3f} nanoseconds"
