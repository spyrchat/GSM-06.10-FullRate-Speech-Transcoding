import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from encoder import RPE_frame_coder
from decoder import RPE_frame_decoder


def process_wav_file(input_file: str, output_file: str):
    # Step 1: Read Input WAV File
    samplerate, data = wavfile.read(input_file)
    data = data.astype(np.int16)

    # Ensure mono audio
    if data.ndim > 1:
        data = data[:, 0]

    # Ensure frame size alignment (multiple of 160)
    frame_size = 160
    if len(data) % frame_size != 0:
        padding_length = frame_size - (len(data) % frame_size)
        data = np.pad(data, (0, padding_length),
                      mode='constant', constant_values=0)

    # Buffers for encoded and reconstructed signals
    encoded_frames = []
    reconstructed_frames = []
    prev_frame_resd = np.zeros(frame_size)  # Initialize with zeros

    # Step 2: Encode Each Frame
    for i in range(0, len(data), frame_size):
        frame = data[i:i+frame_size]

        # Encode frame
        frame_bit_stream, curr_frame_resd = RPE_frame_coder(
            frame, prev_frame_resd)
        encoded_frames.append((frame_bit_stream, curr_frame_resd))

        # Update previous residual for next frame
        prev_frame_resd = curr_frame_resd

    # Step 3: Decode Each Frame
    prev_frame_resd = np.zeros(frame_size)  # Reset residual for decoding
    for frame_bit_stream, curr_frame_resd in encoded_frames:
        reconstructed_frame, prev_frame_resd = RPE_frame_decoder(
            frame_bit_stream, prev_frame_resd)
        reconstructed_frames.append(reconstructed_frame)

    # Step 4: Convert Output to WAV Format
    reconstructed_signal = np.concatenate(
        reconstructed_frames).astype(np.int16)

    # Save the output WAV file
    wavfile.write(output_file, samplerate, reconstructed_signal)

    # Step 5: Plot Input vs. Output Waveforms
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(data, label="Original Signal", color='blue')
    plt.title('Input WAV File')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(reconstructed_signal, label="Reconstructed Signal", color='red')
    plt.title('Output WAV File')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    process_wav_file('ena_dio_tria.wav', 'output.wav')
