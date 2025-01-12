import numpy as np
from scipy.signal import lfilter
from hw_utils import reflection_coeff_to_polynomial_coeff
from typing import Tuple


def RPE_frame_st_decoder(LARc: np.ndarray, curr_frame_st_resd: np.ndarray) -> np.ndarray:
    """
    Short-term decoder for a single frame of voice data.

    :param LARc: np.ndarray - Quantized LAR coefficients.
    :param curr_frame_st_resd: np.ndarray - Prediction residual (d'(n)).
    :return: np.ndarray - Reconstructed voice signal (s0).
    """
    # Step 1: Decode and interpolate LAR coefficients
    A = np.array([20, 20, 20, 20, 13.637, 15, 8.334, 8.824])
    B = np.array([0, 0, -16, -16, -8, -4, -2, -1])
    LAR = (LARc - B) / A

    # Interpolate LAR coefficients over 4 subframes
    interpolated_LAR = np.zeros((4, 8))
    for i in range(8):
        interpolated_LAR[0, i] = 0.75 * LAR[i] + \
            0.25 * LAR[i - 1] if i > 0 else LAR[i]
        interpolated_LAR[1, i] = 0.50 * LAR[i] + \
            0.50 * LAR[i - 1] if i > 0 else LAR[i]
        interpolated_LAR[2, i] = 0.25 * LAR[i] + \
            0.75 * LAR[i - 1] if i > 0 else LAR[i]
        interpolated_LAR[3, i] = LAR[i]

    # Step 2: Convert interpolated LAR to reflection coefficients for each subframe
    reflection_coeffs = np.zeros((4, 8))
    for subframe in range(4):
        for i in range(8):
            if abs(interpolated_LAR[subframe, i]) < 0.675:
                reflection_coeffs[subframe, i] = interpolated_LAR[subframe, i]
            elif 0.675 <= abs(interpolated_LAR[subframe, i]) < 1.225:
                reflection_coeffs[subframe, i] = np.sign(
                    interpolated_LAR[subframe, i]) * (0.5 * abs(interpolated_LAR[subframe, i]) + 0.3375)
            elif abs(interpolated_LAR[subframe, i]) >= 1.225:
                reflection_coeffs[subframe, i] = np.sign(
                    interpolated_LAR[subframe, i]) * (0.125 * abs(interpolated_LAR[subframe, i]) + 0.796875)

    # Step 3: Convert reflection coefficients to LPC coefficients for each subframe
    LPC_coeffs = np.zeros((4, 9))  # 8 coefficients + 1 for the leading 1
    for subframe in range(4):
        LPC_coeffs[subframe, :] = reflection_coeff_to_polynomial_coeff(
            reflection_coeffs[subframe, :])[0]

    # Step 4: Perform short-term synthesis filtering for the entire frame
    s_reconstructed = np.zeros_like(curr_frame_st_resd)
    for subframe in range(4):
        start = subframe * 40
        end = start + 40
        s_reconstructed[start:end] = lfilter(
            [1], LPC_coeffs[subframe, :], curr_frame_st_resd[start:end])

    # Step 5: Post-processing (deemphasis filter)
    beta = 28180 * (2 ** -15)
    s0 = np.zeros_like(s_reconstructed)
    for k in range(len(s_reconstructed)):
        s0[k] = s_reconstructed[k] + beta * (s0[k - 1] if k > 0 else 0)

    return s0


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
                                   n] = curr_subframe_ex[n] + bc[i] * d_double_prime

    # Call the short-term decoder
    s0 = RPE_frame_st_decoder(LARc, reconstructed_residual)

    return s0, reconstructed_residual


def RPE_frame_decoder(
    frame_bit_stream: str,
    prev_frame_resd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decoder for a single frame of voice data.

    :param frame_bit_stream: str - Encoded bitstream for the frame.
    :param prev_frame_resd: np.ndarray - Residual signal from the previous frame.
    :return: Tuple[np.ndarray, np.ndarray] - Reconstructed speech signal and current residual.
    """
    # Perform decoding steps to reconstruct the signal
    # Placeholder: Replace with actual decoder logic
    s0 = np.random.random(160)  # Placeholder signal reconstruction
    curr_frame_resd = np.random.random(160)  # Placeholder residual calculation

    return s0, curr_frame_resd
