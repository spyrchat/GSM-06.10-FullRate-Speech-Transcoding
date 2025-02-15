import numpy as np


def xm_select(e: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Selects the sub-sequence with the maximum energy from the input array.
    This function processes the input array `e` using a predefined filter `H` 
    based on ETSI GSM 06.10 and then divides the filtered signal into interleaved 
    sub-sequences. It calculates the energy of each sub-sequence and selects the one with the maximum energy.
    Args:
        e (np.ndarray): Input array to be processed.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: The selected sub-sequence with the maximum energy.
            - int: The index of the selected sub-sequence.
    """

    xm_samples_length = 13
    subframe_size = len(e)
    H = np.array([-134, -374, 0, 2054, 5741, 8192,
                 5741, 2054, 0, -374, -134]) / (2**13)
    x = np.zeros(len(e))

    for k in range(subframe_size):
        for i in range(11):
            index = k + 5 - i
            if 0 <= index < subframe_size:  # This is the Boundary condition for 3.20
                x[k] += H[i] * e[index]

    x_m_interleved_sequences = []

    for m in range(4):
        x_m = np.zeros(xm_samples_length)
        for i in range(xm_samples_length):
            k = m + 3 * i
            if k < len(x):
                x_m[i] = x[k]
        x_m_interleved_sequences.append(x_m)

    energies = []

    for x_m in x_m_interleved_sequences:
        energy = np.sum(x_m ** 2)
        energies.append(energy)
    energies = np.array(energies)

    # Select the x_m sub-sequence with the maximum energy
    selected_M = np.argmax(energies)
    selected_x_m = x_m_interleved_sequences[selected_M]

    return selected_x_m, selected_M


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
