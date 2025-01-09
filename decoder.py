import numpy as np
from scipy.signal import lfilter
from hw_utils import reflection_coeff_to_polynomial_coeff


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
