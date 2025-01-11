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
    Nc: int,
    bc: float,
    curr_frame_ex_full: np.ndarray,
    curr_frame_st_resd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decoder for short-term and long-term coded frame.

    :param LARc: np.ndarray - Quantized LAR coefficients.
    :param Nc: int - Pitch period from long-term analysis.
    :param bc: float - Gain factor from long-term analysis.
    :param curr_frame_ex_full: np.ndarray - Prediction residual (long-term).
    :param curr_frame_st_resd: np.ndarray - Residual after short-term analysis.
    :return: Tuple[np.ndarray, np.ndarray] - Reconstructed signal and short-term residual.
    """
    # Step 1: Dequantization - LAR to reflection coefficients to LPC
    reflection_coeffs = (np.exp(LARc) - 1) / (np.exp(LARc) + 1)
    a = reflection_coeff_to_polynomial_coeff(reflection_coeffs)

    # Step 2: Long-Term Decoding
    frame_length = 160
    subframe_length = 40
    d_prime = np.zeros(frame_length)

    for i in range(0, frame_length, subframe_length):
        # Current subframe residual
        curr_subframe_ex = curr_frame_ex_full[i:i + subframe_length]
        prev_subframes = d_prime[i:i + subframe_length + 120]

        # Reconstruct d'(n) for the current subframe
        predicted = bc * prev_subframes[Nc:Nc + subframe_length]
        d_prime[i:i + subframe_length] = curr_subframe_ex + predicted

    curr_frame_st_resd = d_prime  # Updated short-term residual

    # Step 3: Short-Term Decoding
    s0 = lfilter([1], [1] + [-ai for ai in a[1:]], d_prime)

    return s0, curr_frame_st_resd
