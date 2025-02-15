from bitstring import BitStream
import numpy as np
from scipy.signal import lfilter
from hw_utils import reflection_coeff_to_polynomial_coeff
from typing import Tuple
from encoder import dequantize_gain_factor, quantize_LAR, decode_LAR, decode_reflection_coeffs


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
    Decode a bitstream into a single frame of audio.
    """
    # Step 1: Parse the bitstream
    bitstream = BitStream(bin=frame_bit_stream)
    LARc = [bitstream.read('int:8')
            for _ in range(8)]  # Read 8 LAR coefficients

    Nc, bc = [], []
    for _ in range(4):  # 4 subframes
        Nc.append(bitstream.read('uint:7'))  # 7 bits for pitch period
        bc.append(bitstream.read('uint:4') / 15.0)  # Scale back gain factor

    # Step 2: Reconstruct the signal
    frame_length = 160
    subframe_length = 40
    reconstructed_residual = np.zeros(frame_length)
    for subframe_start, N, b in zip(range(0, frame_length, subframe_length), Nc, bc):
        prev_residual = prev_frame_resd[max(
            0, subframe_start - 120):subframe_start]
        for n in range(subframe_length):
            reconstructed_residual[subframe_start + n] = b * \
                prev_residual[n - N] if n - N >= 0 else 0

    # Short-term decoding
    s0 = RPE_frame_st_decoder(np.array(LARc), reconstructed_residual)
    return s0, reconstructed_residual
