from bitstring import BitStream, Bits
import numpy as np
from scipy.signal import lfilter
from typing import Tuple
from hw_utils import reflection_coeff_to_polynomial_coeff, polynomial_coeff_to_reflection_coeff


import numpy as np
from scipy.signal import lfilter

import numpy as np


def dequantize_gain_factor(b_c: int) -> float:
    """
    Dequantize the quantized gain factor (b_c) to the corresponding decoded value (b_j').

    :param b_c: int - Quantized gain factor index (0, 1, 2, 3).
    :return: float - Dequantized gain factor (b_j').
    """
    # Quantization levels from Table 3.3
    QLB = [0.10, 0.35, 0.65, 1.00]  # Dequantized values

    return QLB[b_c]


def quantize_gain_factor(b: float) -> int:
    """
    Quantize the gain factor (b) based on defined thresholds.

    :param b: float - The gain factor to quantize.
    :return: float - The quantized gain factor.
    """
    # Decision thresholds and corresponding quantized levels
    DLB = [0.2, 0.5, 0.8]  # Decision Level Boundaries

    # Determine the quantized gain factor
    if b <= DLB[0]:
        return 0
    elif DLB[0] < b <= DLB[1]:
        return 1
    elif DLB[1] < b <= DLB[2]:
        return 2
    elif b > DLB[2]:
        return 3


def decode_reflection_coeffs(LAR_decoded: np.ndarray) -> np.ndarray:
    """
    Decode LAR' coefficients to reflection coefficients r'(i) using GSM 06.10 (equation 3.5).

    :param LAR_decoded: np.ndarray - Decoded Log-Area Ratios (LAR'(i)).
    :return: np.ndarray - Reflection coefficients r'(i).
    """
    reflection_coeffs = np.zeros_like(LAR_decoded)

    for i, LAR_prime in enumerate(LAR_decoded):
        abs_LAR = abs(LAR_prime)
        sign_LAR = np.sign(LAR_prime)  # Sign of LAR'(i)

        if abs_LAR < 0.675:
            reflection_coeffs[i] = LAR_prime  # Case 1: |LAR'(i)| < 0.675
        elif 0.675 <= abs_LAR < 1.225:
            reflection_coeffs[i] = sign_LAR * \
                (0.500 * abs_LAR + 0.3375)  # Case 2
        elif 1.225 <= abs_LAR <= 1.625:
            reflection_coeffs[i] = sign_LAR * \
                (0.125 * abs_LAR + 0.796875)  # Case 3
        else:
            raise ValueError(f"Invalid LAR'(i) value: {
                             LAR_prime}. Expected |LAR'(i)| <= 1.625.")

    return reflection_coeffs


def decode_LAR(LARc: np.ndarray) -> np.ndarray:
    """
    Decode quantized Log-Area Ratios (LAR_C) to LAR'' using predefined A and B values.

    :param LARc: np.ndarray - Quantized Log-Area Ratios (LAR_C).
    :return: np.ndarray - Decoded Log-Area Ratios (LAR'').
    """
    # Constants from GSM 06.10
    A = np.array([20, 20, 20, 20, 13.637, 15, 8.334, 8.824])
    B = np.array([0, 0, -16, -16, -8, -4, -2, -1])

    # Ensure LARc has the correct length
    if len(LARc) != len(A):
        raise ValueError(f"LARc has incorrect length: {
                         len(LARc)}. Expected: {len(A)}")

    # Decode LAR coefficients
    LAR_decoded = (LARc - B) / A

    return LAR_decoded


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

    # Ensure the input LAR has the correct length
    if len(LAR) != len(A):
        raise ValueError(f"LAR has incorrect length: {
                         len(LAR)}. Expected: {len(A)}")

    # Quantization rule
    LARq = np.zeros_like(LAR, dtype=int)
    for i in range(len(LAR)):
        # Validate LAR[i]
        if np.isnan(LAR[i]) or np.isinf(LAR[i]):
            raise ValueError(f"LAR[{i}] is invalid: {LAR[i]}")

        # Apply the linear transformation
        LARq[i] = int(A[i] * LAR[i] + B[i] + 0.5)  # Round to nearest integer

        # Clamp to the valid range
        LARq[i] = max(LAR_min[i], min(LAR_max[i], LARq[i]))

    return LARq


def calculate_LAR(reflection_coeffs: np.ndarray) -> np.ndarray:
    """
    Calculate the Log-Area Ratios (LAR) from reflection coefficients (r(i)).

    :param reflection_coeffs: np.ndarray - Reflection coefficients (r(i)).
    :return: np.ndarray - Log-Area Ratios (LAR).
    """
    LAR = np.zeros_like(reflection_coeffs)

    for i, r in enumerate(reflection_coeffs):
        abs_r = abs(r)
        sign_r = np.sign(r)  # Sign of r(i)

        if abs_r < 0.675:
            LAR[i] = r  # Case 1: |r(i)| < 0.675
        elif 0.675 <= abs_r < 0.950:
            # Case 2: 0.675 <= |r(i)| < 0.950
            LAR[i] = sign_r * (2 * abs_r - 0.675)
        elif 0.950 <= abs_r <= 1.000:
            # Case 3: 0.950 <= |r(i)| <= 1.000
            LAR[i] = sign_r * (8 * abs_r - 6.375)
        else:
            raise ValueError(f"Invalid reflection coefficient r(i) = {
                             r}. Expected |r(i)| <= 1.")

    return LAR


def RPE_frame_st_coder(s0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Short-term coder for a single frame of voice data with LAR quantization, offset compensation, and preemphasis.

    :param s0: np.ndarray - 160 samples of the input voice signal.
    :return: Tuple[np.ndarray, np.ndarray] - Quantized LAR coefficients, Prediction residual.
    """
    # Step 1: Offset Compensation
    alpha = 32735 * (2 ** -15)  # Offset compensation coefficient
    s_offset_compensated = np.zeros(s0.shape)
    for k in range(len(s0)):
        s_offset_compensated[k] = (
            s0[k]
            - (s0[k - 1] if k > 0 else 0)  # Previous input sample
            # Previous output
            + alpha * (s_offset_compensated[k - 1] if k > 0 else 0)
        )

    # Step 2: Preemphasis Filtering
    beta = 28180 * (2 ** -15)  # Preemphasis coefficient
    s_preemphasized = np.zeros(s_offset_compensated.shape)
    for k in range(len(s_offset_compensated)):
        s_preemphasized[k] = (
            s_offset_compensated[k]
            - beta * (s_offset_compensated[k - 1] if k > 0 else 0)
        )
    # Step 3: Compute LPC coefficients (order 8)
    p = 8
    ACF = np.zeros(p + 1)
    for k in range(p + 1):
        ACF[k] = np.sum(s_preemphasized[k:] *
                        s_preemphasized[:len(s_preemphasized) - k])
    R = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            R[i, j] = ACF[np.abs(i - j)]
    r = ACF[1:p + 1]
    w = np.linalg.solve(R, r)

    w_new = [1] + [-wi for wi in w]
    w_new = np.array(w_new)
    # Step 4: Convert LPC coefficients to reflection coefficients and LAR
    reflection_coeffs = polynomial_coeff_to_reflection_coeff(w_new)
    print(reflection_coeffs)
    LAR = calculate_LAR(reflection_coeffs)

    # Step 5: Quantize LAR coefficients using ETSI/GSM rules
    LARc = quantize_LAR(LAR)
    LAR_new = decode_LAR(LARc)
    r_new = decode_reflection_coeffs(LAR_new)
    w_new = reflection_coeff_to_polynomial_coeff(r_new)[0]
    residual = lfilter(w_new, [1], s_preemphasized)

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

        prev_d = np.zeros(120)  # Initialize with zeros
        start_idx = max(0, subframe_start - 120)
        slice_length = subframe_start - start_idx

        # Fill the last `slice_length` elements of `prev_d` with valid data
        if slice_length > 0:
            prev_d[-slice_length:] = prev_frame_st_resd[start_idx:subframe_start]

        # Call RPE_subframe_slt_lte to estimate N and b
        N, b = RPE_subframe_slt_lte(curr_subframe, prev_d)

        # Quantize gain factor (b)
        b_quantized = quantize_gain_factor(b)
        N_quantized = N
        Nc_values_opt.append(N_quantized)
        bc_values_opt.append(b_quantized)
        b_dequantized = dequantize_gain_factor(b_quantized)
        N_dequantized = N_quantized
        # Prediction residual computation
        predicted = np.zeros(subframe_length)
        for n in range(subframe_length):
            if n - N_dequantized >= 0:
                predicted[n] = b_dequantized * \
                    curr_frame_st_resd[subframe_start + n - N_dequantized]
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
