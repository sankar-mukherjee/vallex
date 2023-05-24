from torch.utils.tensorboard import SummaryWriter
from vallex.utils import plot_spectrogram
import matplotlib.pyplot as plt
from pathlib import Path

class TBLogger:
    def __init__(self, log_dir, name):
        self.summary_writer = SummaryWriter(Path(log_dir, name))

    def log_value(self, step, key, val):
        self.summary_writer.add_scalar(key, val, step)
        
    def log_audio(self, step, key, data, sample_rate):
        self.summary_writer.add_audio(tag=key, snd_tensor=data, sample_rate=sample_rate, global_step=step)

    def log_image(self, step, key, img):
        self.summary_writer.add_figure(tag=key, figure=img, global_step=step)

    def log_text(self, step, key, string):
        self.summary_writer.add_text(tag=key, text_string=string, global_step=step)


def init(log_dir, tb_subsets=[]):
    global tb_loggers
    tb_loggers = {s: TBLogger(log_dir, s) for s in tb_subsets}

def log(step, data={}, subset='train'):
    for key, v in data.items():
        tb_loggers[subset].log_value(step, key, v)

def log_audio(step, sample_rate, data={}, subset='train'):
    for key, v in data.items():
        tb_loggers[subset].log_audio(step, key, v, sample_rate)

def log_image(step, data={}, subset='train'):
    for key,v in data.items():
        if 'spectrogram' in key:
            fig = plot_spectrogram(v)
            tb_loggers[subset].log_image(step, key, fig)
            plt.close(fig)
            
def log_text(step, data={}, subset='train'):
    for key, v in data.items():
        tb_loggers[subset].log_text(step, key, v)

def flush():
    for tbl in tb_loggers.values():
        tbl.summary_writer.flush()
