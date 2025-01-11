import numpy as np
from scipy.signal import lfilter
from hw_utils import reflection_coeff_to_polynomial_coeff
from typing import Tuple


def RPE_frame_st_decoder(LARc: np.ndarray, curr_frame_st_resd: np.ndarray) -> np.ndarray:
    """
    Short-term decoder for a single frame of voice data.

    :param LARc: np.ndarray - Quantized LAR coefficients.
    :param curr_frame_st_resd: np.ndarray - Prediction residual (d'(n)).
    :return: np.ndarray - Reconstructed voice signal (s0).
    """
    # Dequantize LAR coefficients
    reflection_coeffs = (np.exp(LARc) - 1) / (np.exp(LARc) + 1)

    # Recompute LPC coefficients
    a = reflection_coeff_to_polynomial_coeff(reflection_coeffs)

    # Synthesize signal
    s_reconstructed = lfilter(
        [1], [1] + [-ai for ai in a[1:]], curr_frame_st_resd)

    return s_reconstructed


def RPE_frame_slt_decoder(
    LARc: np.ndarray,
    Nc: np.ndarray,
    bc: np.ndarray,
    curr_frame_ex_full: np.ndarray
) -> np.ndarray:
    """
    Decoder for a single frame of voice data using short-term and long-term prediction.

    :param LARc: np.ndarray - Quantized LAR coefficients for the frame (8 values).
    :param Nc: np.ndarray - Quantized pitch periods for the 4 subframes.
    :param bc: np.ndarray - Quantized gain factors for the 4 subframes.
    :param curr_frame_ex_full: np.ndarray - Long-term excitation signal for the frame.
    :return: np.ndarray - Reconstructed speech signal for the frame (160 samples).
    """
    frame_length = 160
    subframe_length = 40
    num_subframes = frame_length // subframe_length

    # Step 1: Reconstruct the long-term residual for each subframe
    reconstructed_residual = np.zeros(frame_length)
    for i in range(num_subframes):
        start_idx = i * subframe_length
        end_idx = start_idx + subframe_length
        curr_subframe_ex = curr_frame_ex_full[start_idx:end_idx]
        prev_residual = reconstructed_residual[max(
            0, start_idx - 120):start_idx]

        for n in range(subframe_length):
            d_double_prime = 0
            if n - Nc[i] >= 0:
                d_double_prime = prev_residual[n - Nc[i]]
            reconstructed_residual[start_idx +
                                   n] = curr_subframe_ex[n] + bc[i] * d_double_prime

    # Step 2: Calculate reflection coefficients from LAR coefficients
    reflection_coeffs = np.zeros(8)
    for i in range(8):
        if abs(LARc[i]) < 0.675:
            reflection_coeffs[i] = LARc[i]
        elif 0.675 <= abs(LARc[i]) < 1.225:
            reflection_coeffs[i] = np.sign(
                LARc[i]) * (0.5 * abs(LARc[i]) + 0.3375)
        elif abs(LARc[i]) >= 1.225:
            reflection_coeffs[i] = np.sign(
                LARc[i]) * (0.125 * abs(LARc[i]) + 0.796875)

    # Step 3: Convert reflection coefficients to LPC coefficients
    def reflection_to_lpc(reflection_coeffs):
        order = len(reflection_coeffs)
        a = np.zeros(order + 1)
        a[0] = 1
        for i in range(1, order + 1):
            a[i] = -reflection_coeffs[i - 1]
            for j in range(1, i):
                a[j] += reflection_coeffs[i - 1] * a[i - j]
        return a

    LPC_coeffs = reflection_to_lpc(reflection_coeffs)

    # Step 4: Perform short-term synthesis filtering
    s_reconstructed = lfilter([1], LPC_coeffs, reconstructed_residual)

    # Step 5: Post-processing (deemphasis filter)
    beta = 28180 * (2 ** -15)
    s0 = np.zeros_like(s_reconstructed)
    for k in range(len(s_reconstructed)):
        s0[k] = s_reconstructed[k] + beta * (s0[k - 1] if k > 0 else 0)

    return s0
