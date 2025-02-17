from bitstring import BitStream
import numpy as np
from scipy.signal import lfilter
from hw_utils import reflection_coeff_to_polynomial_coeff, polynomial_coeff_to_reflection_coeff
from typing import Tuple
from encoder import dequantize_gain_factor, decode_LAR, decode_reflection_coeffs
from utils import rpe_dequantize, reconstruct_excitation


def RPE_frame_st_decoder(LARc: np.ndarray, curr_frame_st_resd: np.ndarray) -> np.ndarray:
    """
    Implements Short-Term Synthesis Filtering H_s(z) as per GSM 06.10 (3.2.3),
    using a lattice filter structure.

    :param LARc: np.ndarray - Quantized LAR coefficients (8 values).
    :param curr_frame_st_resd: np.ndarray - Prediction residual signal d'(n).
    :return: np.ndarray - Reconstructed speech signal s_r'(k).
    """
    # Step 1: Convert LARc to reflection coefficients
    decoded_LAR = decode_LAR(LARc)  # Decode LAR to real values
    reflection_coeffs = decode_reflection_coeffs(decoded_LAR)  # Convert to reflection coefficients

    frame_length = len(curr_frame_st_resd)
    order = len(reflection_coeffs)  # Order = 8 for GSM

    # Step 2: Initialize output buffers
    s_r = np.zeros((order + 1, frame_length))
    s_r[0] = curr_frame_st_resd  # First stage is the input residual d'(n)

    # Step 3: Apply recursive lattice filtering
    for i in range(1, order + 1):  # i = 1 to 8 (GSM standard)
        for k in range(frame_length):
            s_r[i, k] = s_r[i - 1, k]  # Copy previous stage output
            if k > 0:
                s_r[i, k] -= reflection_coeffs[order - i] * s_r[i - 1, k - 1]  # Apply reflection coefficient

    # Step 4: Extract final synthesized speech signal s_r'(k)
    s_r_prime = s_r[order]

    # Step 5: Apply Post-Processing (De-emphasis Filtering)
    beta = 28180 * (2 ** -15)
    s_ro = np.zeros(frame_length)
    for k in range(frame_length):
        s_ro[k] = s_r_prime[k] + beta * (s_ro[k - 1] if k > 0 else 0)

    # Step 6: Normalize & Cast to int16 to Prevent Overflow
    s_ro = np.clip(s_ro, -32768, 32767)  # Prevent overflow
    return s_ro.astype(np.int16)  # Cast to int16


def RPE_frame_slt_decoder(
    LARc: np.ndarray,
    Nc: np.ndarray,
    bc: np.ndarray,
    curr_frame_ex_full: np.ndarray,
    curr_frame_st_resd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decoder for a single frame of voice data using short-term and long-term prediction.

    :param LARc: np.ndarray - Quantized LAR coefficients for the frame (8 values).
    :param Nc: np.ndarray - Quantized pitch periods for the 4 subframes.
    :param bc: np.ndarray - Quantized gain factors for the 4 subframes.
    :param curr_frame_ex_full: np.ndarray - Long-term excitation signal for the frame.
    :param curr_frame_st_resd: np.ndarray - Residual signal to be reconstructed.
    :return: Tuple[np.ndarray, np.ndarray] - Reconstructed speech signal (s0) and short-term residual.
    """
    frame_length = 160
    subframe_length = 40
    num_subframes = frame_length // subframe_length

    # Reconstruct the long-term residual for each subframe
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
                                   n] = curr_subframe_ex[n] + dequantize_gain_factor(bc[i]) * d_double_prime

    # Call the short-term decoder
    s0 = RPE_frame_st_decoder(LARc, reconstructed_residual)
    return s0, reconstructed_residual


def RPE_frame_decoder(frame_bit_stream: str, prev_frame_resd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decodes a GSM 06.10 RPE-LTP encoded frame from a bitstream.

    Parameters:
    - frame_bit_stream (str): The binary bitstream representing the encoded frame.
    - prev_frame_resd (np.ndarray): The residual signal from the previous frame (160 samples).

    Returns:
    - Tuple[np.ndarray, np.ndarray]: The reconstructed speech signal and residual signal.
    """

    # Step 1: Parse the bitstream
    bitstream = BitStream(bin=frame_bit_stream) if isinstance(frame_bit_stream, str) else frame_bit_stream

    # Extract LPC coefficients (LARc)
    LARc = np.array([
        bitstream.read('int:6'), bitstream.read('int:6'),
        bitstream.read('int:5'), bitstream.read('int:5'),
        bitstream.read('int:4'), bitstream.read('int:4'),
        bitstream.read('int:3'), bitstream.read('int:3')
    ])

    # Extract subframe parameters
    Nc, bc, Mc, x_maxc, x_mcs = [], [], [], [], []
    num_subframes = 4
    subframe_length = 40

    for _ in range(num_subframes):
        Nc.append(bitstream.read('uint:7'))  # Pitch lag
        bc.append(bitstream.read('uint:2'))  # Gain index
        Mc.append(bitstream.read('uint:2'))  # Grid position
        x_maxc.append(bitstream.read('uint:6'))  # Maximum quantized value
        x_mcs.append(np.array([bitstream.read('uint:3') for _ in range(13)]))  # 13 quantized RPE samples

    # Step 2: Construct the excitation signal
    excitation_signal = np.zeros(160)

    for j in range(num_subframes): 
        subframe_start = j * subframe_length

        # Dequantize RPE samples
        x_ms = rpe_dequantize(x_mcs[j], x_maxc[j])

        # Reconstruct excitation signal using grid position Mc
        excitation = reconstruct_excitation(x_ms, Mc[j])

        # Insert excitation into the full signal
        excitation_signal[subframe_start:subframe_start + subframe_length] += excitation

    # Step 3: Apply Long-Term Synthesis Filtering & Short-Term Synthesis Filtering
    s_ro, reconstructed_residual = RPE_frame_slt_decoder(
        LARc, Nc, bc, excitation_signal, prev_frame_resd
    )

    return s_ro, reconstructed_residual