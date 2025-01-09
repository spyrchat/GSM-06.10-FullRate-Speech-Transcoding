import numpy as np
from scipy.signal import lfilter
from typing import Tuple
from hw_utils import reflection_coeff_to_polynomial_coeff, polynomial_coeff_to_reflection_coeff


def RPE_frame_st_coder(s0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Short-term coder for a single frame of voice data.

    :param s0: np.ndarray - 160 samples of the input voice signal.
    :return: Tuple[np.ndarray, np.ndarray] - Quantized LAR coefficients, Prediction residual.
    """
    # Preprocessing (Offset compensation and pre-emphasis)
    s_preemphasized = lfilter([1, -0.9375], [1], s0)  # Pre-emphasis filter

    # Compute LPC coefficients (order 8)
    autocorr = np.correlate(s_preemphasized, s_preemphasized, mode='full')
    autocorr = autocorr[len(s_preemphasized) - 1:]  # Keep only positive lags
    lpc_order = 8
    R = autocorr[:lpc_order + 1]
    a, e_final = reflection_coeff_to_polynomial_coeff(R[1:] / R[0])

    # Convert LPC coefficients to reflection coefficients and LAR
    reflection_coeffs = polynomial_coeff_to_reflection_coeff(a)
    LARc = np.log((1 + reflection_coeffs) / (1 - reflection_coeffs))

    # Quantize LAR coefficients (placeholder for actual quantization logic)
    LARc_quantized = np.round(LARc, decimals=2)  # Example quantization

    # Calculate residual d'(n)
    prediction = lfilter([1] + [-ai for ai in a[1:]], [1], s_preemphasized)
    residual = s_preemphasized - prediction

    return LARc_quantized, residual
