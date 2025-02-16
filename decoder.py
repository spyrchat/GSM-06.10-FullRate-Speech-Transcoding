from bitstring import BitStream
import numpy as np
from scipy.signal import lfilter
from hw_utils import reflection_coeff_to_polynomial_coeff
from typing import Tuple
from encoder import dequantize_gain_factor, decode_LAR, decode_reflection_coeffs
from utils import rpe_dequantize, reconstruct_excitation


def RPE_frame_st_decoder(LARc: np.ndarray, curr_frame_st_resd: np.ndarray) -> np.ndarray:
    """
    Short-term decoder for a single frame of voice data.

    :param LARc: np.ndarray - Quantized LAR coefficients.
    :param curr_frame_st_resd: np.ndarray - Prediction residual (d'(n)).
    :return: np.ndarray - Reconstructed voice signal (s0).
    """
    LAR_new = decode_LAR(LARc)
    r_new = decode_reflection_coeffs(LAR_new)
    w_new = reflection_coeff_to_polynomial_coeff(r_new)[0]

    s = lfilter([1], w_new, curr_frame_st_resd)

    beta = 28180 * (2 ** -15)
    sro = np.zeros(s.shape)
    for k in range(len(s)):
        sro[k] = s[k] + beta * (sro[k - 1] if k > 0 else 0)
    sro = sro.astype(np.int16)
    return sro


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


def RPE_frame_decoder(frame_bit_stream: str, prev_frame_resd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decodes a GSM 06.10 RPE-LTP encoded frame from a bitstream.

    Parameters:
    - frame_bit_stream (str): The binary bitstream representing the encoded frame.
    - prev_frame_resd (np.ndarray): The residual signal from the previous frame (160 samples).

    Returns:
    - Tuple[np.ndarray, np.ndarray]: The reconstructed speech signal (s0) and residual signal.
    """

    # Step 1: Parse the bitstream
    bitstream = BitStream(bin=frame_bit_stream) if isinstance(
        frame_bit_stream, str) else frame_bit_stream

    # 1a. Extract LPC coefficients (LARc) - used once per frame
    LARc = [
        bitstream.read('int:6'), bitstream.read('int:6'),
        bitstream.read('int:5'), bitstream.read('int:5'),
        bitstream.read('int:4'), bitstream.read('int:4'),
        bitstream.read('int:3'), bitstream.read('int:3')
    ]

    # 1b. Extract subframe parameters
    Nc, bc, Mc, x_maxc, x_mcs = [], [], [], [], []
    num_subframes = 4
    subframe_length = 40

    for _ in range(num_subframes):
        Nc.append(bitstream.read('uint:7'))  # Pitch lag
        bc.append(bitstream.read('uint:2'))  # Gain index
        Mc.append(bitstream.read('uint:2'))  # Grid position
        x_maxc.append(bitstream.read('uint:6'))  # Maximum quantized value
        x_mcs.append(np.array([bitstream.read('uint:3')
                     for _ in range(13)]))  # 13 quantized RPE samples

    # Step 2: Construct the excitation signal
    excitation_signal = np.zeros(160)

    for j in range(num_subframes): 
        subframe_start = j * subframe_length

        # Dequantize RPE samples
        x_ms = rpe_dequantize(x_mcs[j], x_maxc[j])

        # Reconstruct excitation signal using grid position Mc
        excitation = reconstruct_excitation(x_ms, Mc[j])

        # Insert excitation into the full signal
        excitation_signal[subframe_start:subframe_start +
                          subframe_length] += excitation

    # Step 3: Call RPE_frame_slt_decoder to perform full synthesis
    s0, reconstructed_residual = RPE_frame_slt_decoder(
        LARc, Nc, bc, excitation_signal, prev_frame_resd
    )

    return s0, reconstructed_residual
