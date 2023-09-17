import random

import librosa.display
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# from utils import audio, text
# from params.params import Params as hp


class Logger_tensorboard:
    """Static class wrapping methods for Tensorboard logging and plotting."""

    @staticmethod
    def initialize(logdir, flush_seconds):
        """Initialize Tensorboard logger.

        Arguments:
            logdir -- location of Tensorboard log files
            flush_seconds -- see Tensorboard documentation
        """
        Logger_tensorboard._sw = SummaryWriter(log_dir=logdir, flush_secs=flush_seconds)

    @staticmethod
    def progress(progress, prefix='', length=70):
        """Prints a pretty console progress bar.

        Arguments:
            progress -- percentage (from 0 to 1.0)
        Keyword argumnets:
            prefix (default: '') -- string which is prepended to the progress bar
            length (default: 70) -- size of the full-size bar
        """
        progress *= 100
        step = 100/length
        filled, reminder = int(progress // step), progress % step
        loading_bar = filled * '█'
        loading_bar += '░' if reminder < step / 3 else '▒' if reminder < step * 2/3 else '▓'
        loading_bar += max(0, length - filled) * '░' if progress < 100 else ''
        print(f'\r{prefix} {loading_bar} {progress:.1f}%', end=('' if progress < 100 else '\n'), flush=True)

    @staticmethod
    def training(train_step, losses, acc, learning_rate,duration,loadtime):
        """Log batch training.
        
        Arguments:
            train_step -- number of the current training step
            losses (dictionary of {loss name, value})-- dictionary with values of batch losses
            gradient (float) -- gradient norm
            learning_rate (float) -- current learning rate
            duration (float) -- duration of the current step
            classifier (float) -- accuracy of the reversal classifier
        """  

        # log losses
        total_loss = sum(losses.values())
        Logger_tensorboard._sw.add_scalar(f'Train/loss_total', total_loss, train_step)
        for n, l in losses.items():
            Logger_tensorboard._sw.add_scalar(f'Train/loss_{n}', l, train_step)  

        # log duration
        Logger_tensorboard._sw.add_scalar("Train/duration", duration, train_step)

        # log loadtime
        Logger_tensorboard._sw.add_scalar("Train/loadtime", loadtime, train_step)

        # log acc
        for n, l in acc.items():
            Logger_tensorboard._sw.add_scalar(f'Train/acc_{n}', l, train_step) 
        
        # log learning rate
        Logger_tensorboard._sw.add_scalar("Train/learning_rate", learning_rate, train_step)


    @staticmethod
    def evaluation(train_step, losses, acer, acc, learning_rate,duration,loadtime):
        """Log evaluation results.
        
        Arguments:
            train_step -- number of the current training step
            losses (dictionary of {loss name, value})-- dictionary with values of batch losses
            gradient (float) -- gradient norm
            learning_rate (float) -- current learning rate
            duration (float) -- duration of the current step
            classifier (float) -- accuracy of the reversal classifier
        """  

        # log losses
        total_loss = sum(losses.values())
        Logger_tensorboard._sw.add_scalar(f'eval/loss_total', total_loss, train_step)
        for n, l in losses.items():
            Logger_tensorboard._sw.add_scalar(f'eval/loss_{n}', l, train_step)  

        # log duration
        Logger_tensorboard._sw.add_scalar("Train/duration", duration, train_step)

        # log loadtime
        Logger_tensorboard._sw.add_scalar("Train/loadtime", loadtime, train_step)

        # log acc
        for n, l in acc.items():
            Logger_tensorboard._sw.add_scalar(f'eval/acc_{n}', l, train_step) 

        # log acer
        for n, l in acer.items():
            Logger_tensorboard._sw.add_scalar(f'eval/acer_{n}', l, train_step) 
        
        # log learning rate
        Logger_tensorboard._sw.add_scalar("eval/learning_rate", learning_rate, train_step)