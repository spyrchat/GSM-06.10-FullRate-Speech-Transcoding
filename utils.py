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


def rpe_quantize(rpe_samples: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Quantizes the Regular Pulse Excitation (RPE) sequence based on ETSI GSM 06.10.

    Parameters:
    - rpe_samples (np.ndarray): The input RPE sequence to be quantized.

    Returns:
    - tuple:
        - np.ndarray: The 3-bit quantized normalized RPE samples.
        - int: The index of the quantized maximum amplitude.
    """

    # Step 1: Determine the maximum absolute value of the RPE sequence
    max_amplitude = np.max(np.abs(rpe_samples))

    # Step 2: Quantize max_amplitude using predefined intervals (from Table 3.5)
    quantization_intervals = np.array([
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

    # Find the closest quantized value and its corresponding index
    quantized_max_index = np.searchsorted(
        quantization_intervals, max_amplitude, side="right") - 1
    quantized_max_value = quantization_intervals[quantized_max_index]

    # Step 3: Normalize RPE samples
    normalized_samples = (rpe_samples / quantized_max_value).astype(float)

    # Step 4: Quantize normalized samples using predefined 3-bit levels (from Table 3.6)
    quantization_levels = np.array([
        -28672, -20480, -12288, -4096,  4096, 12288, 20480, 28672
    ]) / (2**15)  # Scale to match normalization in dequantization

    quantized_samples = np.zeros_like(normalized_samples, dtype=int)

    for i, sample in enumerate(normalized_samples):
        quantized_samples[i] = np.searchsorted(
            quantization_levels, sample, side="right") - 1

    return quantized_samples, quantized_max_index


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
