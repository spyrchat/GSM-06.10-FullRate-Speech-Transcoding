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
    Implements quantization of b gain.

    Parameters:
    - b: float, gain

    Returns:
    - bc: int, quantized gain
    """
    DLB = [0.2, 0.5, 0.8]
    
    if b <= DLB[0]:
        return 0
    elif DLB[0] < b <= DLB[1]:
        return 1
    elif DLB[1] < b <= DLB[2]:
        return 2
    elif DLB[2] < b:
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
    A = [20.0, 20.0, 20.0, 20.0, 13.637, 15.0, 8.334, 8.824]
    B = [0.0, 0.0, 4.0, -5.0, 0.184, -3.5, -0.666, -2.235]

    # Ensure LARc has the correct length
    if len(LARc) != len(A):
        raise ValueError(f"LARc has incorrect length: {
                         len(LARc)}. Expected: {len(A)}")

    # Decode LAR coefficients
    LAR_decoded = (LARc - B) / A

    return LAR_decoded


def quantize_LAR(LAR: np.ndarray) -> np.ndarray:
    """
    Implements quantization of LAR coefficients to LARc.

    Parameters:
    - LAR: np.ndarray, Log-Area Ratios (LAR)

    Returns:
    - LARc: np.ndarray, quantized LAR coefficients
    """
    A = np.array([20, 20, 20, 20, 13.637, 15, 8.334, 8.824])
    B = [0.0, 0.0, 4.0, -5.0, 0.184, -3.5, -0.666, -2.235]
    LARc = np.zeros_like(LAR, dtype=int)
    for i in range(len(LAR)):
        quantized = A[i] * LAR[i] + B[i]
        LARc[i] = int(quantized + np.sign(quantized) * 0.5)  # Round to nearest integer

        # Clamp to specified range
        if i < 2:
            LARc[i] = max(-32, min(31, LARc[i]))
        elif i < 4:
            LARc[i] = max(-16, min(15, LARc[i]))
        elif i < 6:
            LARc[i] = max(-8, min(7, LARc[i]))
        else:
            LARc[i] = max(-4, min(3, LARc[i]))
    return LARc



def calculate_LAR(reflection_coeffs: np.ndarray) -> np.ndarray:
    """
    Implements the transformation to convert reflection coefficients r to LAR.

    Parameters:
    - ra: np.ndarray, reflection coefficients r

    Returns:
    - LAR: np.ndarray, Log-Area Ratios (LAR)
    """
    LAR = np.zeros_like(reflection_coeffs, dtype=float)
    for i in range(len(LAR)):
        abs_r = abs(reflection_coeffs[i])
        if abs_r < 0.675:
            LAR[i] = reflection_coeffs[i]
        elif 0.675 <= abs_r < 0.950:
            LAR[i] = np.sign(reflection_coeffs[i]) * (2 * abs_r - 0.675)
        elif 0.950 <= abs_r <= 1.000:
            LAR[i] = np.sign(reflection_coeffs[i]) * (8 * abs_r - 6.375)

    return LAR


def xm_select(input_signal: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Selects the sub-sequence with the maximum energy from the input signal.

    This function processes the input signal using a predefined FIR filter `H` 
    based on ETSI GSM 06.10 and then divides the filtered signal into interleaved 
    sub-sequences. It calculates the energy of each sub-sequence and selects the one with the maximum energy.

    Parameters:
    - input_signal (np.ndarray): Input signal to be processed.

    Returns:
    - selected_subsequence (np.ndarray): The selected sub-sequence with the highest energy.
    - selected_index (int): The index of the selected sub-sequence.
    """

    num_samples = 13  # Length of each sub-sequence
    signal_length = len(input_signal)  # Length of the input signal

    # FIR Filter `H` for pre-processing (as per ETSI GSM 06.10)
    fir_filter = np.array([-134, -374, 0, 2054, 5741, 8192,
                           5741, 2054, 0, -374, -134]) / (2**13)

    # Apply FIR filter to the input signal
    filtered_signal = np.zeros(signal_length)

    for k in range(signal_length):
        for i in range(len(fir_filter)):
            index = k + 5 - i  # Center-aligned filtering
            if 0 <= index < signal_length:  # Boundary check
                filtered_signal[k] += fir_filter[i] * input_signal[index]

    # Divide filtered signal into interleaved sub-sequences
    interleaved_subsequences = []

    for sub_index in range(4):  # 4 possible sub-sequences
        subsequence = np.zeros(num_samples)
        for i in range(num_samples):
            sample_index = sub_index + 3 * i  # Interleaved sampling
            if sample_index < len(filtered_signal):
                subsequence[i] = filtered_signal[sample_index]
        interleaved_subsequences.append(subsequence)

    # Compute energy of each sub-sequence
    subsequence_energies = np.array(
        [np.sum(subseq ** 2) for subseq in interleaved_subsequences])

    # Select the sub-sequence with the maximum energy
    selected_index = np.argmax(subsequence_energies)
    selected_subsequence = interleaved_subsequences[selected_index]

    return selected_subsequence, selected_index


def rpe_quantize(x_ms: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Quantizes the RPE sequence as described in the ETSI GSM 06.10 standard.

    Parameters:
    - x_ms: np.ndarray, the RPE sequence to be quantized

    Returns:
    - x_mcs: np.ndarray, quantized normalized RPE samples
    - x_maxc: int, quantized x_max
    """
    # Step 1: Find maximum absolute value x_max
    x_max = np.max(np.abs(x_ms))

    # Step 2: Quantize x_max
    x_max_intervals = [
        (0, 31), (32, 63), (64, 95), (96, 127), (128,
                                                 159), (160, 191), (192, 223), (224, 255),
        (256, 287), (288, 319), (320, 351), (352,
                                             383), (384, 415), (416, 447), (448, 479),
        (480, 511), (512, 575), (576, 639), (640,
                                             703), (704, 767), (768, 831), (832, 895),
        (896, 959), (960, 1023), (1024, 1151), (1152,
                                                1279), (1280, 1407), (1408, 1535),
        (1536, 1663), (1664, 1791), (1792, 1919), (1920,
                                                   2047), (2048, 2303), (2304, 2559),
        (2560, 2815), (2816, 3071), (3072, 3327), (3328,
                                                   3583), (3584, 3839), (3840, 4095),
        (4096, 4607), (4608, 5119), (5120, 5631), (5632,
                                                   6143), (6144, 6655), (6656, 7167),
        (7168, 7679), (7680, 8191), (8192, 9215), (9216,
                                                   10239), (10240, 11263), (11264, 12287),
        (12288, 13311), (13312, 14335), (14336,
                                         15359), (15360, 16383), (16384, 18431),
        (18432, 20479), (20480, 22527), (22528,
                                         24575), (24576, 26623), (26624, 28671),
        (28672, 30719), (30720, 32767)
    ]

    x_maxc = None
    x_max_ = None

    for i, interval in enumerate(x_max_intervals):
        if interval[0] <= x_max < interval[1] + 1:
            x_maxc = i
            x_max_ = interval[1]
            break

    # Step 3: Normalize x_m by x_max_
    xs = (x_ms / x_max_).astype(float)

    # Step 4: Quantize normalized samples using Table 3.6
    x_mcs = np.zeros_like(xs, dtype=int)
    for i, x in enumerate(xs):
        if -32768 / (2**15) <= x < -24576 / (2**15):
            x_mcs[i] = 0
        elif -24576 / (2**15) <= x < -16384 / (2**15):
            x_mcs[i] = 1
        elif -16384 / (2**15) <= x < -8192 / (2**15):
            x_mcs[i] = 2
        elif -8192 / (2**15) <= x < 0:
            x_mcs[i] = 3
        elif 0 <= x < 8192 / (2**15):
            x_mcs[i] = 4
        elif 8192 / (2**15) <= x < 16384 / (2**15):
            x_mcs[i] = 5
        elif 16384 / (2**15) <= x < 24576 / (2**15):
            x_mcs[i] = 6
        elif 24576 / (2**15) <= x <= 32767 / (2**15):
            x_mcs[i] = 7

    return x_mcs, x_maxc


def rpe_dequantize(quantized_samples: np.ndarray, quantized_max_index: int) -> np.ndarray:
    """
    Dequantizes the Regular Pulse Excitation (RPE) sequence based on ETSI GSM 06.10.

    Parameters:
    - quantized_samples (np.ndarray): The 3-bit quantized RPE sequence.
    - quantized_max_index (int): The index corresponding to the quantized maximum amplitude.

    Returns:
    - np.ndarray: The reconstructed (dequantized) RPE sequence.
    """

    # Step 1: Lookup table for normalized quantization levels (from Table 3.6)
    quantization_levels = np.array([
        -28672, -20480, -12288, -4096,  4096, 12288, 20480, 28672
    ]) / (2**15)  # Scale to match normalization in quantization

    # Convert quantized sample indices to corresponding normalized values
    normalized_samples = quantization_levels[quantized_samples]

    # Step 2: Lookup table for block maximum values (from Table 3.5)
    block_max_values = np.array([
        31, 63, 95, 127, 159, 191, 223, 255,
        287, 319, 351, 383, 415, 447, 479, 511,
        575, 639, 703, 767, 831, 895, 959, 1023,
        1151, 1279, 1407, 1535, 1663, 1791, 1919,
        2047, 2303, 2559, 2815, 3071, 3327, 3583,
        3839, 4095, 4607, 5119, 5631, 6143, 6655,
        7167, 7679, 8191, 9215, 10239, 11263, 12287,
        13311, 14335, 15359, 16383, 18431, 20479, 22527,
        24575, 26623, 28671, 30719, 32767
    ])

    # Retrieve the corresponding block maximum value
    dequantized_max = block_max_values[quantized_max_index]

    # Step 3: Rescale the normalized values back to their original range
    dequantized_samples = normalized_samples * dequantized_max

    return dequantized_samples


def reconstruct_excitation(dequantized_rpe_seq: np.ndarray, grid_position: int) -> np.ndarray:
    """
    Reconstructs the excitation signal based on the ETSI GSM 06.10 standard.

    Parameters:
    - dequantized_rpe_seq (np.ndarray): The dequantized RPE sequence.
    - grid_position (int): The dequantized grid position M.

    Returns:
    - np.ndarray: The reconstructed excitation signal (40 samples).
    """

    # Step 1: Initialize the excitation signal with zeros
    excitation_signal = np.zeros(40)

    # Step 2: Place dequantized RPE samples at intervals of 3 in the excitation signal
    for i, sample in enumerate(dequantized_rpe_seq):
        index = grid_position + 3 * i
        excitation_signal[index] = sample

    return excitation_signal
