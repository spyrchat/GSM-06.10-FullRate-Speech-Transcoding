# 📢 GSM 06.10 Full-Rate Speech Codec

## 🔥 Overview
This project implements a **GSM 06.10 Full-Rate Speech Codec** based on the **Regular Pulse Excitation (RPE-LTP)** algorithm. The codec is capable of **encoding** and **decoding** speech audio, transforming it into a compressed bitstream and reconstructing it back to speech.

## 🚀 Features
- ✅ **Short-Term Linear Predictive Coding (LPC)** for speech signal modeling
- ✅ **Long-Term Prediction (LTP)** for pitch tracking and residual analysis
- ✅ **Regular Pulse Excitation (RPE)** for efficient speech compression
- ✅ **Bitstream Formation (260-bit frames)** for transmission
- ✅ **WAV file support** for input and output
- ✅ **Visualization of results** via Matplotlib

## 📜 GSM 06.10 Standard
GSM 06.10 is a **speech compression standard** used in digital cellular systems. It processes audio in **160-sample frames (20ms at 8kHz sampling rate)** and achieves **full-rate speech encoding** while maintaining quality suitable for telecommunication applications.

## 📁 Project Structure
```
📂 GSM-06.10-Codec
├── 📄 README.md        # This file
├── 📜 encoder.py       # Encoding process (LPC, LTP, RPE)
├── 📜 decoder.py       # Decoding process (inverse LPC, excitation reconstruction)
├── 📜 utils.py         # Helper functions for signal processing
├── 📜 hw_utils.py      # Utility functions for coefficient transformations
├── 📜 demo1.py        # Basic encoding/decoding demonstration
├── 📜 demo2.py        # Extended demonstration with enhanced features
├── 📜 demo3.py        # Full feature demonstration and performance evaluation
└── 🎵 sample.wav       # Example input audio file (not included in repository)
```

## 🛠 Installation & Setup
### 📌 Prerequisites
Make sure you have **Python 3.x** and the required dependencies installed:
```sh
pip install numpy scipy matplotlib bitstring
```

### 📌 Running the Encoder & Decoder
To process an audio file using the GSM 06.10 codec:
```sh
python demo3.py
```
This will:
1. Read the input `.wav` file
2. Encode it using GSM 06.10
3. Decode it back to a `.wav` file
4. Plot the original vs reconstructed waveform

## 🎯 How It Works
### **1️⃣ Short-Term Analysis**
- Uses **Linear Predictive Coding (LPC)** to model speech
- Converts LPC coefficients to **Log-Area Ratios (LAR)**
- Encodes LAR using **quantization**

### **2️⃣ Long-Term Prediction**
- Estimates **pitch period (N) and gain factor (b)**
- Predicts residuals based on prior speech frames

### **3️⃣ Regular Pulse Excitation (RPE)**
- Selects **sub-sequence with maximum energy**
- **Quantizes excitation signal**
- **Encodes into a 260-bit frame**

### **4️⃣ Decoding Process**
- **Dequantizes & reconstructs excitation signal**
- **Applies inverse filtering** to retrieve speech
- **Synthesizes final speech waveform**


## 📜 References
- [GSM 06.10 Standard (ETSI)](https://www.etsi.org/deliver/etsi_en/300900_300999/300961/08.01.01_60/en_300961v080101p.pdf)
- [Python Signal Processing - SciPy](https://docs.scipy.org/doc/scipy/reference/signal.html)



