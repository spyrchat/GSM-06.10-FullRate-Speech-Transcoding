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


def quantize_RPE_sequence(selected_x_m: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Quantizes the RPE sequence as described in the ETSI GSM 06.10 standard.
    Args:
        x_ms (np.ndarray): The RPE sequence to be quantized.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: The quantized normalized RPE samples.
            - int: The quantized normalized RPE samples.
    """

    x_max = np.max(np.abs(selected_x_m))

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
    xs = (selected_x_m / x_max_).astype(float)

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
