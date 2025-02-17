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

    sro = np.zeros_like(s)
    sro[0] = s[0]
    beta = 28180 * (2 ** -15)
    for k in range(1, len(s)):
        sro[k] = s[k] + beta * sro[k - 1]

    return sro.astype(np.int16)


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
            reconstructed_residual[start_idx + n] = curr_subframe_ex[n] + \
                dequantize_gain_factor(bc[i]) * d_double_prime

    # Call the short-term decoder
    s0 = RPE_frame_st_decoder(LARc, reconstructed_residual)
    return s0, reconstructed_residual


def RPE_frame_decoder(frame_bitstream: str, prev_residual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decodes a GSM 06.10 RPE-LTP encoded frame from a bitstream.

    Parameters:
    - frame_bitstream (str or BitStream): The encoded frame bitstream.
    - prev_residual (np.ndarray): The residual signal from the previous frame (160 samples).

    Returns:
    - reconstructed_signal (np.ndarray): Reconstructed speech signal (160 samples).
    - curr_residual (np.ndarray): Updated residual signal for the current frame (160 samples).
    """

    num_subframes = 4
    subframe_size = 40

    # Ensure frame_bitstream is a BitStream object
    if isinstance(frame_bitstream, str):
        frame_bitstream = BitStream(bin=frame_bitstream)

    # Step 1: Parse bitstream to extract parameters
    quantized_lar = np.zeros(8, dtype=int)
    pitch_lags = np.zeros(num_subframes, dtype=int)
    gain_indices = np.zeros(num_subframes, dtype=int)
    excitation_signal = np.zeros(160, dtype=float)

    # Extract quantized LAR coefficients from bitstream
    quantized_lar[0] = frame_bitstream.read('int:6')
    quantized_lar[1] = frame_bitstream.read('int:6')
    quantized_lar[2] = frame_bitstream.read('int:5')
    quantized_lar[3] = frame_bitstream.read('int:5')
    quantized_lar[4] = frame_bitstream.read('int:4')
    quantized_lar[5] = frame_bitstream.read('int:4')
    quantized_lar[6] = frame_bitstream.read('int:3')
    quantized_lar[7] = frame_bitstream.read('int:3')

    for subframe_idx in range(num_subframes):
        # Extract pitch lag (Nc) and gain index (bc) from bitstream
        pitch_lags[subframe_idx] = frame_bitstream.read('uint:7')
        gain_indices[subframe_idx] = frame_bitstream.read('uint:2')

        # Extract Mc, x_maxc, and x_mcs
        selected_index = frame_bitstream.read('uint:2')
        quantized_max_index = frame_bitstream.read('uint:6')
        quantized_subseq = np.array(
            [frame_bitstream.read('uint:3') for _ in range(13)], dtype=int
        )

        # Dequantize x_mcs and Mc
        dequantized_subseq = rpe_dequantize(
            quantized_subseq, quantized_max_index)

        # Reconstruct excitation signal
        excitation = reconstruct_excitation(dequantized_subseq, selected_index)
        excitation_signal[(subframe_idx * subframe_size)
                           :((subframe_idx + 1) * subframe_size)] = excitation

    # Step 2: Decode using RPE_frame_slt_decoder
    reconstructed_signal, curr_residual = RPE_frame_slt_decoder(
        quantized_lar, pitch_lags, gain_indices, excitation_signal, prev_residual
    )

    return reconstructed_signal, curr_residual
