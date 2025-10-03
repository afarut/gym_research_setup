from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from logger.base import LoggerBase


class TensorBoardLogger(LoggerBase):
    def __init__(self, log_dir):
        super().__init__(log_dir)

        self.writer = SummaryWriter(log_dir=log_dir+"/tensorboard")
        self.steps = defaultdict(int)

    def log(self, metrics: dict, prefix=""):
        for key, value in metrics.items():
            self.writer.add_scalar(prefix + key, value, self.steps[prefix + key])
            self.steps[prefix + key] += 1
        self.writer.flush()
    
    def close(self):
        self.writer.close()