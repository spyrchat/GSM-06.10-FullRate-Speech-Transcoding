import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from encoder import RPE_frame_coder
from decoder import RPE_frame_decoder


def process_wav_file(input_file: str, output_file: str):
    samplerate, data = wavfile.read(input_file)
    data = data.astype(np.int16)
    # Ensure mono audio for simplicity
    if data.ndim > 1:
        data = data[:, 0]

    # Buffers for residuals and output
    prev_frame_resd = np.zeros(160, dtype=np.float32)
    reconstructed_signal = []

    for i in range(0, len(data), 160):
        frame = data[i:i+160]
        if len(frame) < 160:
            frame = np.pad(frame, (0, 160 - len(frame)), 'constant')

        # Encode and decode the frame
        frame_bit_stream, curr_frame_resd = RPE_frame_coder(
            frame, prev_frame_resd)
        reconstructed_frame, prev_frame_resd = RPE_frame_decoder(
            frame_bit_stream, prev_frame_resd)

        reconstructed_signal.extend(reconstructed_frame)

    # Convert back to int16 for WAV compatibility
    reconstructed_signal = np.clip(
        reconstructed_signal, -32768, 32767).astype(np.int16)

    # Save the output file
    wavfile.write(output_file, samplerate, reconstructed_signal)

    # Plot the waveforms
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(data)
    plt.title('Input WAV File')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(reconstructed_signal)
    plt.title('Output WAV File')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
