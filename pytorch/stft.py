import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from torch.nn.parameter import Parameter

class STFT(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True, in_chans=2):
        """Modified STFT for multichannel input"""
        super(STFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode
        self.in_chans = in_chans  # Number of input channels

        # By default, use the entire frame
        if win_length is None:
            win_length = n_fft

        # Set the default hop, if it's not already specified
        if hop_length is None:
            hop_length = int(win_length // 4)

        fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

        # Pad the window out to n_fft size
        fft_window = librosa.util.pad_center(fft_window, n_fft)

        def dft_matrix(self, n_fft):
            """Create the DFT matrix for STFT computation."""
            omega = np.exp(-2 * np.pi * 1j / n_fft)
            j, k = np.meshgrid(np.arange(n_fft), np.arange(n_fft))
            W = np.power(omega, j * k)
            return W
        # DFT & IDFT matrix
        self.W = dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        # Create conv layers for real and imaginary parts for each channel
        self.conv_real = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=out_channels,
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1,
            groups=1, bias=False) for _ in range(in_chans)])

        self.conv_imag = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=out_channels,
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1,
            groups=1, bias=False) for _ in range(in_chans)])




        # Set weights for all conv layers
        for conv in self.conv_real:
            conv.weight.data = torch.Tensor(
                np.real(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[None, :, :]
        for conv in self.conv_imag:
            conv.weight.data = torch.Tensor(
                np.imag(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[None, :, :]

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, channels, data_length)"""
        if self.center:
            input = F.pad(input, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real_parts = []
        imag_parts = []
        for i in range(self.in_chans):
            real = self.conv_real[i](input[:, i:i+1, :])  # Process each channel separately
            imag = self.conv_imag[i](input[:, i:i+1, :])
            real_parts.append(real)
            imag_parts.append(imag)

        # Combine the results from different channels
        real = torch.cat(real_parts, dim=1)
        imag = torch.cat(imag_parts, dim=1)

        return real, imag


class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
                 window='hann', center=True, pad_mode='reflect', power=2.0,
                 freeze_parameters=True):
        """Calculate spectrogram using pytorch. The STFT is implemented with
        Conv1d. The function has the same output of librosa.core.stft
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window, center=center,
                         pad_mode=pad_mode, freeze_parameters=True, in_chans=2)

    def forward(self, input):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)
        Returns:
          spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real ** 2 + imag ** 2

        return spectrogram


class LogmelFilterBank(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=64, fmin=50, fmax=14000, is_log=True,
                 ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        """Calculate logmel spectrogram using pytorch. The mel filter bank is
        the pytorch implementation of as librosa.filters.mel
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                                        fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, channels, time_steps)

        Output: (batch_size, time_steps, mel_bins)
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output

    def power_to_db(self, input):
        """Power to db, this function is the pytorch implementation of
        librosa.core.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        # if self.top_db is not None:
        #     if self.top_db < 0:
        #         raise ParameterError('top_db must be non-negative')
        #     log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db, max=np.inf)

        return log_spec


# # Example usage
# # Create an instance of modified STFT for 2 channels
# stft = STFT(in_chans=2)
#
# # Create a dummy input tensor representing a batch of audio with 2 channels (batch_size=2, channels=2, data_length=5000)
# dummy_input = torch.randn(2, 2, 5000)
#
# # Compute the STFT
# real, imag = stft(dummy_input)
#
# # Check the shape of the outputs
# print(real.shape, imag.shape)  # Expected shapes: (2, 2, n_fft // 2 + 1, time_steps)
