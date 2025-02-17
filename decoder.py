from bitstring import BitStream, ReadError
import numpy as np
from scipy.signal import lfilter
from hw_utils import reflection_coeff_to_polynomial_coeff, polynomial_coeff_to_reflection_coeff
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
            reconstructed_residual[start_idx +
                                   n] = curr_subframe_ex[n] + dequantize_gain_factor(bc[i]) * d_double_prime

    # Call the short-term decoder
    s0 = RPE_frame_st_decoder(LARc, reconstructed_residual)
    return s0, reconstructed_residual


from bitstring import BitStream, ReadError
import numpy as np

def RPE_frame_decoder(frame_bit_stream: str, prev_frame_resd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decodes a GSM 06.10 RPE-LTP encoded frame from a bitstream.

    Parameters:
    - frame_bit_stream (str or BitStream): The encoded frame bitstream.
    - prev_frame_resd (np.ndarray): The residual signal from the previous frame (160 samples).

    Returns:
    - s0: np.ndarray, reconstructed speech signal (160 samples).
    - curr_frame_resd: np.ndarray, updated residual signal for the current frame (160 samples).
    """

    num_subframes = 4
    subframe_size = 40

    try:
        # Ensure frame_bit_stream is a BitStream object
        if isinstance(frame_bit_stream, str):
            frame_bit_stream = BitStream(bin=frame_bit_stream)

        # Step 1: Parse bitstream to extract parameters
        LARc = np.zeros(8, dtype=int)
        Ncs = np.zeros(num_subframes, dtype=int)
        bcs = np.zeros(num_subframes, dtype=int)
        curr_frame_ex_full_ = np.zeros(160, dtype=float)

        # Extract LARc coefficients from bitstream
        LARc[0] = frame_bit_stream.read('int:6')
        LARc[1] = frame_bit_stream.read('int:6')
        LARc[2] = frame_bit_stream.read('int:5')
        LARc[3] = frame_bit_stream.read('int:5')
        LARc[4] = frame_bit_stream.read('int:4')
        LARc[5] = frame_bit_stream.read('int:4')
        LARc[6] = frame_bit_stream.read('int:3')
        LARc[7] = frame_bit_stream.read('int:3')

        for j in range(num_subframes):
            # Extract Nc (pitch lag) and bc (gain factor) from bitstream
            Ncs[j] = frame_bit_stream.read('uint:7')
            bcs[j] = frame_bit_stream.read('uint:2')

            # Extract Mc, x_maxc, and x_mcs
            Mc = frame_bit_stream.read('uint:2')
            x_maxc = frame_bit_stream.read('uint:6')
            x_mcs = np.array([frame_bit_stream.read('uint:3') for _ in range(13)], dtype=int)

            # Dequantize x_mcs and Mc
            x_ms_ = rpe_dequantize(x_mcs, x_maxc)
            M_ = Mc

            # Reconstruct e_
            e_ = reconstruct_excitation(x_ms_, M_)
            curr_frame_ex_full_[(j * subframe_size):((j + 1) * subframe_size)] = e_

        # Step 2: Decode using RPE_frame_slt_decoder
        s0, curr_frame_resd = RPE_frame_slt_decoder(LARc, Ncs, bcs, curr_frame_ex_full_, prev_frame_resd)

        return s0, curr_frame_resd

    except ReadError:
        raise ValueError("Error reading the bitstream. Ensure the bitstream is properly formatted.")
