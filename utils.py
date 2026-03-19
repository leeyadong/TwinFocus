
from matplotlib import pyplot as plt
import os
import matplotlib.pyplot as plt
import json
from pathlib import Path
import torch
import numpy as np
import math, torch, torch.nn.functional as F, torch.fft as fft
C0 = 299_792_458.0

def zero_pad_range_fft(adc, pad_factor):

	if pad_factor == 1:
		return np.fft.fft(adc, axis=-1)
	Nz        = adc.shape[-1]
	Nz_pad = Nz * pad_factor
	pad_width = [(0, 0)] * adc.ndim
	pad_width[-1] = (0, Nz_pad - Nz)
	adc_padded = np.pad(adc, pad_width, mode='constant')

	return np.fft.fft(adc_padded, axis=-1)


def sar_bp_conv_block(
		adc,
		pos_txrx,
		config,
		target_range,
		scene_size,
		pixels_num,
		num_bins=5,
		sim_err=False,
		pad_factor=4,
		frame_block=1024,
		randseed = 2025,
		error = 0.5,
		exp = "size",
		device='cuda'):

	pos_tx    = torch.from_numpy(pos_txrx[0]).to(device)
	pos_tx[:,0] *= -1
	pos_rx = torch.from_numpy(pos_txrx[1]).to(device)
	pos_rx[:, 0] *= -1
	src = zero_pad_range_fft(adc, pad_factor)
	src = torch.from_numpy(src.reshape((-1, src.shape[-1]))).to(device)

	Nf, Nz = src.shape

	fs, slope, fc = (config[k] for k in
					 ["Sampling_Rate_sps", "chirpSlope", "startFrequency"])

	# ---- range & bin bookkeeping ------------------------------------
	f_b    = torch.arange(Nz, device=device) * fs / Nz
	r_axis = (C0 / (2.0 * slope)) * f_b
	idx_c  = torch.argmin(torch.abs(r_axis - target_range))
	half   = num_bins // 2
	sel    = torch.arange(idx_c-half, idx_c+half+1, device=device).clamp(0, Nz-1)
	k_r    = 4.0 * math.pi * (fc + f_b[sel]) / C0      # (#bins,)

	# ---- imaging grid (1, ny, nx, 3) -------------------------------
	nx, ny = pixels_num
	x = torch.linspace(*scene_size[0], nx, device=device)
	y = torch.linspace(*scene_size[1], ny, device=device)
	Y, X = torch.meshgrid(y, x, indexing='ij')
	tgt  = torch.stack([X, Y, torch.full_like(X, target_range)], dim=-1).unsqueeze(0)
	img = torch.zeros((ny, nx), dtype=torch.complex64, device=device)
	np.random.seed(randseed)
	phase_sets = []
	# ---- iterate over frame *blocks* -------------------------------
	for b_idx, f0 in enumerate(range(0, Nf, frame_block)):

		f1 = min(f0 + frame_block, Nf)
		F  = f1 - f0
		partial_img = torch.zeros_like(img)
		phase_set = torch.zeros((F, ny, nx), dtype=torch.complex64, device=device)
		pos_tx_blk  = pos_tx[f0:f1]
		pos_rx_blk = pos_rx[f0:f1]
		sig_blk  = src[f0:f1][:, sel]

		# ---- 4a) optional position-error injection -------------------
		if sim_err:
			z_err = np.ones((F, 1)) * np.random.normal(loc=0, scale=error) * 1e-3
			z_err_t = torch.from_numpy(z_err).to(device)
			pos_tx_blk[:, 2:3] += z_err_t
			pos_rx_blk[:, 2:3] += z_err_t

		diff_tx = tgt - pos_tx_blk[:, None, None, :]
		diff_rx = tgt - pos_rx_blk[:, None, None, :]
		R = (torch.linalg.norm(diff_tx, dim=-1) + torch.linalg.norm(diff_rx, dim=-1))/2

		for b, k in enumerate(k_r):
			ph = torch.exp(-1j * k * R)
			phase_set += sig_blk[:, b][:, None, None] * ph
			partial_img += (sig_blk[:, b][:, None, None] * ph).sum(dim=0)

		img += partial_img
		phase_sets.append(phase_set)

		del R, sig_blk, pos_tx_blk, pos_rx_blk
		torch.cuda.empty_cache()

	if frame_block==86:
		return img, torch.stack(phase_sets, dim=-1)
	else:
		return img, None

def to_torch(img_np: np.ndarray, device: str = "cuda") -> torch.Tensor:
	img = torch.from_numpy(img_np).float()
	if img.ndim == 3:
		img = img.abs().sum(-1)  # (H,W)
	img = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
	return img.to(device)


def focus(scan: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
	phase = torch.exp(1j * phi).view(1, 1, 1, -1)
	img = (scan * phase).sum(-1).sum(0)
	return soft_thr(img)


def soft_thr(img):
	img = img.abs().unsqueeze(0).unsqueeze(0)
	img = img / img.max()
	img = torch.sigmoid(1 * (img - 0.4))
	return img


def convert_multi2mono(multi_data, config, target_range):
	antenna_azimuthonly = list(map(lambda x: x - 1, config['ChannelOrder']))
	F0 = config["startFrequency"]
	c = C0
	lambda_ = c / F0
	Fs = config["Sampling_Rate_sps"]
	Ks = config["Slope_calib"]
	n_sample = config["numSamplePerChirp"]
	adc_start = 5e-6  # adc start sampling offset
	f0 = F0 + adc_start * Ks  # start frequency
	f = f0 + np.arange(n_sample) * Ks / Fs  # wideband frequency
	kw = 2 * np.pi * f / c  # wideband wavenumber

	mono_data = multi2mono(multi_data, kw, lambda_, target_range, antenna_azimuthonly)

	return mono_data

def multi2mono(multi_data, kw, lambda_, target_range, antenna_azimuthonly):
	"""
	This function converts the multistatic radar data to monostatic radar data

	:param multi_data: the multistatic radar data
	:param kw: the wideband wavenumber
	:param lambda_: the wavelength of the transmiting signal
	:param target_range: the distance(m) of the imaging plane away from the array
	:param antenna_azimuthonly: the MIMO antenna positions of TI cascaded radar
	:return the converted monostatic radar data
	"""

	kw = kw.reshape(1, 1, -1)

	# Calculate antenna distance
	yAxisRx = np.concatenate((np.arange(0, 4, 1), np.arange(11, 15, 1), np.arange(46, 54, 1)), axis=0) * lambda_ / 2
	rxAntPos = np.column_stack((np.zeros(16), yAxisRx, np.zeros(16)))

	dx_TxRx = lambda_ * 17
	dy_TxRx = lambda_ * 8
	yAxisTx = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 9, 10, 11]) * lambda_ / 2
	xAxisTx = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 6]) * (-lambda_ / 2)
	txAntPos = np.column_stack((dx_TxRx + xAxisTx, dy_TxRx + yAxisTx, np.zeros(12)))

	nRx, _ = rxAntPos.shape
	nTx, _ = txAntPos.shape
	txT = txAntPos.reshape(nTx, 1, -1)
	rxT = rxAntPos.reshape(1, nRx, -1)
	virtualChPos = txT + rxT
	virtualChPos = np.reshape(np.transpose(virtualChPos, (1, 0, 2)), (-1, 3))

	distAntennas = txT - rxT
	distAntennas = np.reshape(np.transpose(distAntennas, (1, 0, 2)), (-1, 3), order='F')
	distAntennas = distAntennas[antenna_azimuthonly, :]

	dx_r = distAntennas[:, 0]
	dy_r = distAntennas[:, 1]

	# Phase correction for multistatic
	phase_correction = np.exp(-1j * kw * (dx_r[:, np.newaxis]**2 + dy_r[:, np.newaxis]**2) / (4 * target_range))
	#phase_correction = np.transpose(phase_correction, (1, 0, 2))

	mono_data = multi_data * (phase_correction)

	return mono_data


