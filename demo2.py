import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
from encoder import RPE_frame_coder, RPE_frame_slt_coder
from decoder import RPE_frame_decoder, RPE_frame_slt_decoder


def process_wav_file(input_file: str, output_file: str):
    samplerate, data = wavfile.read(input_file)
    data = data.astype(np.int16)
    # Ensure mono audio for simplicity
    if data.ndim > 1:
        data = data[:, 0]

    # Ensure the length of the data is an integer multiple of 160
    frame_size = 160
    if len(data) % frame_size != 0:
        # Calculate required padding
        padding_length = frame_size - (len(data) % frame_size)
        data = np.pad(data, (0, padding_length),
                      mode='constant', constant_values=0)

    # Buffers for residuals and output
    encoded_signal = []
    reconstructed_signal = []
    prev_frame_st_resd = np.zeros(160)

    for i in range(0, len(data) // 160):
        frame = data[i*160:(i+1)*160]
        # Encode and decode the frame
        LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd = RPE_frame_slt_coder(
            frame, prev_frame_st_resd)

        encoded_signal.append(
            (LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd))

    for i in range(0, len(data) // 160):
        reconstructed_frame, _ = RPE_frame_slt_decoder(
            *encoded_signal[i])

        reconstructed_signal.append(reconstructed_frame)
    # Convert back to int16 for WAV compatibility
    # reconstructed_signal = np.clip(
    #     reconstructed_signal, -32768, 32767).astype(np.int16)
    reconstructed_signal = np.concat(reconstructed_signal).astype(np.int16)
    print(reconstructed_signal.shape)

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


if __name__ == '__main__':
    process_wav_file('ena_dio_tria.wav', 'demo_2_output.wav')
