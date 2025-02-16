from bitstring import BitStream, Bits
import numpy as np
from scipy.signal import lfilter
from typing import Tuple
from hw_utils import reflection_coeff_to_polynomial_coeff, polynomial_coeff_to_reflection_coeff
from utils import rpe_quantize, xm_select, quantize_gain_factor, dequantize_gain_factor, quantize_LAR, decode_LAR, decode_reflection_coeffs, calculate_LAR

import numpy as np
from scipy.signal import lfilter

import numpy as np


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


def RPE_frame_coder(input_speech_frame: np.ndarray, prev_residual_signal: np.ndarray) -> Tuple[str, np.ndarray]:
    """
    Encodes a frame using the RPE frame coder.

    Parameters:
    - input_speech_frame (np.ndarray): Input speech signal (160 samples).
    - prev_residual_signal (np.ndarray): Residual signal from the previous frame (160 samples).

    Returns:
    - frame_bitstream (str): The 260-bit binary representation of the encoded frame.
    - current_residual_signal (np.ndarray): The residual signal for the current frame (160 samples).
    """
    num_subframes = 4
    subframe_length = 40

    # Step 1: Perform short-term and long-term encoding using RPE_frame_slt_coder
    encoded_LARc, pitch_lags, gain_indices, excitation_signal, current_residual_signal = RPE_frame_slt_coder(
        input_speech_frame, prev_residual_signal
    )

    # Step 2: Initialize the bitstream
    frame_bitstream = BitStream()

    # Append LAR coefficients to the bitstream
    frame_bitstream.append(f'int:6={encoded_LARc[0]}')
    frame_bitstream.append(f'int:6={encoded_LARc[1]}')
    frame_bitstream.append(f'int:5={encoded_LARc[2]}')
    frame_bitstream.append(f'int:5={encoded_LARc[3]}')
    frame_bitstream.append(f'int:4={encoded_LARc[4]}')
    frame_bitstream.append(f'int:4={encoded_LARc[5]}')
    frame_bitstream.append(f'int:3={encoded_LARc[6]}')
    frame_bitstream.append(f'int:3={encoded_LARc[7]}')

    # Step 3: Process each subframe
    for subframe_idx in range(num_subframes):
        # Append pitch lag (Nc) to the bitstream
        frame_bitstream.append(f'uint:7={pitch_lags[subframe_idx]}')

        # Append gain index (bc) to the bitstream
        frame_bitstream.append(f'uint:2={gain_indices[subframe_idx]}')

        # Extract subframe excitation signal
        subframe_start = subframe_idx * subframe_length
        subframe_excitation = excitation_signal[subframe_start: subframe_start + subframe_length]

        # Select x_m sub-sequence with the highest energy
        selected_subsequence, selected_index = xm_select(subframe_excitation)

        # Quantize the selected sub-sequence
        quantized_subsequence, quantized_max_index = rpe_quantize(
            selected_subsequence)

        # Append sub-sequence selection index (Mc) and max value quantization index (x_maxc) to the bitstream
        frame_bitstream.append(f'uint:2={selected_index}')
        frame_bitstream.append(f'uint:6={quantized_max_index}')

        # Append the quantized samples to the bitstream
        for quantized_sample in quantized_subsequence:
            frame_bitstream.append(f'uint:3={quantized_sample}')

    return frame_bitstream.bin, current_residual_signal
