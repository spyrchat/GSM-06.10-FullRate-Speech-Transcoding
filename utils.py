from scipy.io import wavfile
from encoder import RPE_frame_coder
from decoder import RPE_frame_decoder
import numpy as np


def process_wav_file(input_file: str, output_file: str):
    """
    Process a .wav file by encoding and decoding it in 160-sample chunks.

    :param input_file: str - Path to the input .wav file.
    :param output_file: str - Path to save the processed .wav file.
    """
    # Read the .wav file
    samplerate, data = wavfile.read(input_file)

    # Ensure mono audio for simplicity
    if len(data.shape) > 1:
        data = data[:, 0]

    # Prepare buffers
    prev_frame_resd = np.zeros(160)
    reconstructed_data = []

    # Process the file in 160-sample chunks
    for i in range(0, len(data), 160):
        s0 = data[i:i+160]
        if len(s0) < 160:
            s0 = np.pad(s0, (0, 160 - len(s0)), 'constant')

        # Encode the frame
        frame_bit_stream, curr_frame_resd = RPE_frame_coder(
            s0, prev_frame_resd)

        # Decode the frame
        reconstructed_frame, curr_frame_resd = RPE_frame_decoder(
            frame_bit_stream, prev_frame_resd)

        # Append the reconstructed frame to the output
        reconstructed_data.extend(reconstructed_frame)

        # Update the previous residual
        prev_frame_resd = curr_frame_resd

    # Convert reconstructed data to np.int16 format
    reconstructed_data = np.array(reconstructed_data, dtype=np.int16)

    # Save the reconstructed .wav file
    wavfile.write(output_file, samplerate, reconstructed_data)
