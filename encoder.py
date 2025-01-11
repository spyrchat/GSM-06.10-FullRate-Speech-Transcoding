import numpy as np
from scipy.signal import lfilter
from typing import Tuple
from hw_utils import reflection_coeff_to_polynomial_coeff, polynomial_coeff_to_reflection_coeff


import numpy as np
from scipy.signal import lfilter


def quantize_LAR(LAR: np.ndarray) -> np.ndarray:
    """
    Quantize the Log-Area Ratios (LAR) coefficients based on the ETSI/GSM 06.10 rules.

    :param LAR: np.ndarray - Array of LAR coefficients.
    :return: np.ndarray - Quantized LAR coefficients.
    """
    # Table values for A[i] and B[i] based on the standard
    A = np.array([20, 20, 20, 20, 13.637, 15, 8.334, 8.824])
    B = np.array([0, 0, -16, -16, -8, -4, -2, -1])
    LAR_min = np.array([-32, -32, -16, -16, -8, -4, -2, -1])
    LAR_max = np.array([31, 31, 15, 15, 7, 3, 1, 0])

    # Quantization rule
    LARq = np.zeros_like(LAR, dtype=int)
    for i in range(len(LAR)):
        # Apply the linear transformation
        LARq[i] = int(A[i] * LAR[i] + B[i] + 0.5)  # Round to nearest integer

        # Clamp to the valid range
        LARq[i] = max(LAR_min[i], min(LAR_max[i], LARq[i]))

    return LARq


def RPE_frame_st_coder(s0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Short-term coder for a single frame of voice data with LAR quantization.

    :param s0: np.ndarray - 160 samples of the input voice signal.
    :return: Tuple[np.ndarray, np.ndarray] - Quantized LAR coefficients, Prediction residual.
    """
    # Step 1: Preprocessing (Offset compensation and pre-emphasis)
    s_preemphasized = lfilter([1, -0.9375], [1], s0)  # Pre-emphasis filter

    # Step 2: Compute LPC coefficients (order 8)
    autocorr = np.correlate(s_preemphasized, s_preemphasized, mode='full')
    autocorr = autocorr[len(s_preemphasized) - 1:]  # Keep only positive lags
    lpc_order = 8
    R = autocorr[:lpc_order + 1]
    a, e_final = reflection_coeff_to_polynomial_coeff(R[1:] / R[0])

    # Step 3: Convert LPC coefficients to reflection coefficients and LAR
    reflection_coeffs = polynomial_coeff_to_reflection_coeff(a)
    LAR = np.log((1 + reflection_coeffs) / (1 - reflection_coeffs))

    # Step 4: Quantize LAR coefficients using ETSI/GSM rules
    LARc = quantize_LAR(LAR)

    # Step 5: Calculate residual d'(n)
    prediction = lfilter([1] + [-ai for ai in a[1:]], [1], s_preemphasized)
    residual = s_preemphasized - prediction

    return LARc, residual


def RPE_frame_slt_coder(
    s0: np.ndarray,
    prev_frame_st_resd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Short-term and long-term coder for a single frame of voice data with quantized LTP gain.

    :param s0: np.ndarray - 160 samples of the input voice signal.
    :param prev_frame_st_resd: np.ndarray - Residual from the previous frame (160 samples).
    :return: Tuple containing:
        - LARc: Quantized LAR coefficients.
        - Nc: Estimated pitch period.
        - bc: Quantized gain factors for all subframes.
        - curr_frame_ex_full: Full prediction residual for the current frame.
        - curr_frame_st_resd: Residual after short-term analysis.
    """
    # Step 1: Short-Term Analysis (reuse existing function)
    LARc, curr_frame_st_resd = RPE_frame_st_coder(s0)

    # Step 2: Long-Term Analysis
    frame_length = 160
    subframe_length = 40
    Nc_values = range(40, 121)  # Pitch range (as per ETSI)
    curr_frame_ex_full = np.zeros_like(curr_frame_st_resd)
    bc_quantized = []
    Nc_values_opt = []

    # Decision levels (DLB) and quantized levels (b_c)
    DLB = [0.2, 0.5, 0.8]
    b_c = [0.1, 0.35, 0.65, 1.0]

    for i in range(0, frame_length, subframe_length):
        # Current subframe
        curr_subframe = curr_frame_st_resd[i:i + subframe_length]
        prev_subframes = prev_frame_st_resd[i:i + subframe_length + 120]

        # Pitch period estimation (maximize cross-correlation)
        cross_corr = [
            np.sum(curr_subframe * prev_subframes[j:j + subframe_length])
            for j in Nc_values
        ]
        Nc = Nc_values[np.argmax(cross_corr)]
        Nc_values_opt.append(Nc)

        # Gain factor estimation
        b_num = np.sum(curr_subframe * prev_subframes[Nc:Nc + subframe_length])
        b_den = np.sum(prev_subframes[Nc:Nc + subframe_length] ** 2)
        b = b_num / b_den if b_den != 0 else 0.0

        # Quantize b using decision levels and quantized levels
        if b < DLB[0]:
            bc = b_c[0]
        elif b < DLB[1]:
            bc = b_c[1]
        elif b < DLB[2]:
            bc = b_c[2]
        else:
            bc = b_c[3]
        bc_quantized.append(bc)

        # Prediction residual computation
        predicted = bc * prev_subframes[Nc:Nc + subframe_length]
        residual = curr_subframe - predicted
        curr_frame_ex_full[i:i + subframe_length] = residual

    return np.array(LARc), Nc_values_opt, np.array(bc_quantized), curr_frame_ex_full, curr_frame_st_resd
