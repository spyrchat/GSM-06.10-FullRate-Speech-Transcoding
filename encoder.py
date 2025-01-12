from bitstring import BitStream, Bits
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


def RPE_subframe_slt_lte(
    d: np.ndarray,
    prev_d: np.ndarray
) -> Tuple[int, float]:
    """
    Estimate the pitch period (N) and gain factor (b) for a subframe.

    :param d: np.ndarray - Current subframe residual signal (40 samples).
    :param prev_d: np.ndarray - Previous residual signal (120 samples).
    :return: Tuple[int, float] - Estimated pitch period (N) and gain factor (b).
    """
    # Validate inputs
    if len(d) != 40:
        raise ValueError("Current subframe d must have 40 samples.")
    if len(prev_d) != 120:
        raise ValueError("Previous residual prev_d must have 120 samples.")

    # Pitch period search range
    pitch_min = 40
    pitch_max = 120

    # Step 1: Compute cross-correlation for the pitch period range
    cross_corr = []
    for N in range(pitch_min, pitch_max + 1):
        # Compute cross-correlation between d and delayed prev_d
        corr = np.sum(d * prev_d[-N:][:40])  # Correlation for delay N
        cross_corr.append(corr)

    # Step 2: Find the pitch period N with the maximum cross-correlation
    N_opt = np.argmax(cross_corr) + pitch_min  # Add pitch_min to offset index

    # Step 3: Calculate the gain factor b
    num = np.sum(d * prev_d[-N_opt:][:40])  # Numerator: energy of alignment
    den = np.sum(prev_d[-N_opt:][:40] ** 2)  # Denominator: energy of prev_d
    b_opt = num / den if den != 0 else 0.0  # Prevent division by zero

    return N_opt, b_opt


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
        - Nc: Estimated pitch periods for all subframes.
        - bc: Quantized gain factors for all subframes.
        - curr_frame_ex_full: Full prediction residual for the current frame.
        - curr_frame_st_resd: Residual after short-term analysis.
    """
    # Step 1: Short-Term Analysis (reuse provided function)
    LARc, curr_frame_st_resd = RPE_frame_st_coder(s0)

    # Step 2: Long-Term Analysis
    frame_length = 160
    subframe_length = 40
    Nc_values_opt = []
    bc_values_opt = []
    curr_frame_ex_full = np.zeros(frame_length)

    for subframe_start in range(0, frame_length, subframe_length):
        curr_subframe = curr_frame_st_resd[subframe_start:subframe_start + subframe_length]

        prev_d = np.zeros(120)  # Ensure 120 samples
        start_idx = max(0, subframe_start - 120)
        end_idx = subframe_start
        prev_d[-(end_idx - start_idx):] = prev_frame_st_resd[start_idx:end_idx]

        # Call RPE_subframe_slt_lte to estimate N and b
        N, b = RPE_subframe_slt_lte(curr_subframe, prev_d)

        # Quantize gain factor (b)
        DLB = [0.2, 0.5, 0.8]
        b_c = [0.1, 0.35, 0.65, 1.0]
        if b < DLB[0]:
            b_quantized = b_c[0]
        elif b < DLB[1]:
            b_quantized = b_c[1]
        elif b < DLB[2]:
            b_quantized = b_c[2]
        else:
            b_quantized = b_c[3]

        Nc_values_opt.append(N)
        bc_values_opt.append(b_quantized)

        # Prediction residual computation
        predicted = np.zeros(subframe_length)
        for n in range(subframe_length):
            if n - N >= 0:
                predicted[n] = b_quantized * \
                    curr_frame_st_resd[subframe_start + n - N]
            else:
                predicted[n] = 0.0
        residual = curr_subframe - predicted
        curr_frame_ex_full[subframe_start:subframe_start +
                           subframe_length] = residual

    return np.array(LARc), np.array(Nc_values_opt), np.array(bc_values_opt), curr_frame_ex_full, curr_frame_st_resd


def RPE_frame_coder(s0: np.ndarray, prev_frame_resd: np.ndarray) -> Tuple[str, np.ndarray]:
    """
    Encode a single frame into a bitstream using bitstring.
    """
    # Use RPE_frame_slt_coder for short-term and long-term analysis
    LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd = RPE_frame_slt_coder(
        s0, prev_frame_resd)

    # Step 3: Create the bitstream
    bitstream = BitStream()
    for lar in LARc:
        bitstream.append(f"int:8={lar}")  # 8 bits for LAR coefficients

    for n, b in zip(Nc, bc):
        bitstream.append(f"uint:7={n}")   # 7 bits for pitch period
        # 4 bits for quantized gain factor index
        bitstream.append(f"uint:4={int(b * 15)}")

    return bitstream.bin, curr_frame_st_resd
