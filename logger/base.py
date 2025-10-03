class LoggerBase:
    def __init__(self, log_dir, *args, **kwargs):
        self.log_dir = log_dir
    
    def log(self, metrics):
        raise NotImplementedError