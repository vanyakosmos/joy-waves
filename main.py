import argparse
import os
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv
from matplotlib.patches import Polygon
from numpy.core.multiarray import ndarray
from scipy.signal import filtfilt, get_window


class ReadMode(Enum):
    GRAY = auto()
    SAT = auto()  # saturation


class JoyWave(object):
    def __init__(self, **kwargs):
        self.read_mode = ReadMode.GRAY
        # If true: dark colors -> waves, light colors -> flat lines
        self.dark = True
        # Use with binary. If th > 0 then mask out values that less then th.
        self.threshold = 0.2
        # If true: ignore color grades and use binary threshold instead.
        self.binary = self.threshold != 0
        # Smoothing level, intuitively can be interpreted as wave length.
        self.wave_len = 4
        # Rescale factor, intuitively can be interpreted as reversed wave height.
        self.wave_height = 10

        # Number of lines that should be skipped. Control density of lines.
        self.skip_lines = 2
        # Bigger == more details from image.
        self.resize_height = 500
        # Height of final image.
        self.canvas_height = 2000

        self.noise_level = 0.05
        self.noise_smooth = 4

        self.edge_win = None  # type: ndarray

        self.update(**kwargs)
        self.validate()

    def update(self, **kwargs):
        self.dark = kwargs.get('dark', self.dark)
        self.threshold = kwargs.get('threshold', self.threshold)
        self.binary = self.threshold != 0
        self.wave_len = kwargs.get('wave_len', self.wave_len)
        self.wave_height = kwargs.get('wave_height', self.wave_height)

        self.skip_lines = kwargs.get('skip_lines', self.skip_lines)
        self.resize_height = kwargs.get('resize_height', self.resize_height)
        self.canvas_height = kwargs.get('canvas_height', self.canvas_height)

        self.read_mode = kwargs.get('read_mode', self.read_mode.name)
        self.read_mode = ReadMode[self.read_mode]

        self.noise_level = kwargs.get('noise_level', self.noise_level)
        self.noise_smooth = kwargs.get('noise_smooth', self.noise_smooth)

        self.edge_win = None

    def validate(self):
        assert self.skip_lines >= 0, \
            "Can't skip less then 0 lines."
        assert self.binary and self.threshold != 0 or not self.binary, \
            "In binary mode threshold should be specified."

    def get_image(self, im_path: str):
        im = Image.open(im_path)

        # if image has transparent layer - replace it with white background
        im = im.convert('RGBA')
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        bg.paste(im, mask=im)
        im = bg
        # read as greyscale or as rgb adn then extract saturation
        if self.read_mode == ReadMode.SAT:
            im = im.convert('RGB')
        else:
            im = im.convert('L')
        # normalize size
        h, w = im.size
        s = self.resize_height
        if s > 0:
            im = im.resize((int(s * h / w), s))
        arr = np.array(im, dtype=np.float)
        # obtain saturation values
        if self.read_mode == ReadMode.SAT:
            arr = rgb_to_hsv(arr / 255)[:, :, 1]
        # bound between 0.01 and 0.99
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = np.clip(arr, 0.01, 0.99)
        arr = arr[::-1]
        return arr

    def filter(self, signal: ndarray, length) -> ndarray:
        return filtfilt(np.ones(length) / length, [1.], signal) * self.edge_win

    def exponentiate(self, signal: ndarray) -> ndarray:
        h = self.wave_height
        return h * (np.exp(signal / h) - 1)

    def reverse_row(self, row):
        if self.dark:
            row = (1.0 - row)
        return row

    def generate_signal(self, row: ndarray) -> ndarray:
        if self.binary and self.threshold != 0:
            signal = np.zeros_like(row)
            if self.threshold > 0:
                mask = row > self.threshold
            else:
                mask = row < -self.threshold
            ones = np.ones_like(row[mask])
            signal[mask] = np.random.chisquare(ones)
        else:
            ones = np.ones_like(row)
            signal = np.random.chisquare(ones)
        return signal

    def normalize_signal(self, signal: ndarray, row: ndarray) -> ndarray:
        if not self.binary:
            signal = signal * row
        noise = self.noise_level * np.random.chisquare(np.ones_like(row))
        noise = self.filter(noise, self.noise_smooth)

        if self.wave_len >= 2:
            signal = self.filter(signal, self.wave_len)
        else:
            signal = signal * self.edge_win

        signal = self.exponentiate(signal + noise)
        return signal

    def get_signal(self, row: ndarray) -> ndarray:
        row = self.reverse_row(row)
        signal = self.generate_signal(row)
        signal = self.normalize_signal(signal, row)
        return signal

    def save_fig(self, fig, ax, path: str):
        ax.autoscale_view()
        ax.axis('tight')
        ax.axis('off')
        fig.savefig(path, facecolor='k', edgecolor='k')

    def generate(self, im_path: str, res_path: str = None, **kwargs):
        if not res_path:
            res_path, ext = os.path.splitext(im_path)
            res_path = f'{res_path}-wavy.png'
        self.update(**kwargs)
        self.validate()

        im = self.get_image(im_path)
        H, W = im.shape
        # need to smooth edges because with spike on the edge polygon patches will jump around
        self.edge_win = get_window('hamming', 17)[:8]
        self.edge_win = np.r_[self.edge_win, np.ones(W - 16), self.edge_win[::-1]]

        size = self.canvas_height // 100
        fig = plt.figure(1, figsize=(size, int(size * H / W)))
        fig.patch.set_facecolor('black')
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        x = np.arange(W)

        for row_index in range(H):
            if row_index % (self.skip_lines + 1) != 0:
                continue
            y = self.get_signal(im[row_index])
            ax.add_patch(Polygon(
                np.c_[x, row_index + 2 * y],
                fc='k', ec='0.85', lw=0.5,
                closed=False, zorder=-row_index
            ))
            if row_index % 10 == 0:
                print(f'{row_index} / {H}', end='\r')
        print(f'saving...', end='\r')
        self.save_fig(fig, ax, res_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+',
                        help="Paths to images.")

    parser.add_argument('--no-d', dest='dark', action='store_false')
    parser.add_argument('-d', '--d', dest='dark', action='store_true',
                        help="Choose dark side. If true: dark colors -> waves, "
                             "light colors -> flat lines. Default: True")

    parser.add_argument('-t', dest='threshold', type=float, default=0.2,
                        help="Binary threshold. If 0 then don't use threshold. "
                             "Default: 0.2")

    parser.add_argument('--wl', dest='wave_len', type=int, default=4,
                        help="Signal smoothing parameter. Can be interpreted as wave length. "
                             "Default: 4")
    parser.add_argument('--wh', dest='wave_height', type=float, default=10,
                        help="Signal amplitude scaling. "
                             "Can be interpreted as reversed wave height. "
                             "Default: 10")

    parser.add_argument('--sl', dest='skip_lines', type=int, default=2,
                        help="Number of lines to skip. Default: 2")
    parser.add_argument('--rh', dest='resize_height', type=int, default=0,
                        help="Resize height. Bigger - more info preserved from original image. "
                             "No sense to make it bigger than original height. "
                             "If less then or equal to 0 - use original image height. "
                             "Default: 0")
    parser.add_argument('--ch', dest='canvas_height', type=int, default=2000,
                        help="Canvas height. Height of result image. Default: 2000")

    parser.add_argument('--rm', dest='read_mode', default='GRAY', choices=['GRAY', 'SAT'],
                        help="Read image in greyscale or as RGB and get saturation after that. "
                             "Default: GRAY")

    parser.add_argument('--nl', dest='noise_level', type=float, default=0.05,
                        help="Noise level. Default: 0.05")
    parser.add_argument('--ns', dest='noise_smooth', type=int, default=4,
                        help="Noise smoothing factor. Default: 4")

    args = parser.parse_args()
    gen = JoyWave()
    for im_path in args.images:
        gen.generate(im_path, **vars(args))


if __name__ == '__main__':
    main()
