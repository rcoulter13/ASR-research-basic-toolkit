# ASR-research-basic-toolkit
Python scripts for basic Automatic Speech Recognition tasks such as finding the SNR, noising data phonetically for Grammar Error Checking (seq2seq) training, easy data augmentation, easy KenLM creation, etc.

### *Contents*
* Get Average SNR for File of Audio Files
* Phonetic Noising Script for (GEC usage)
* JSON Formatter for Huggingface Seq2Seq (GEC usage)
* Audio Data Augmentation
* KenLM ARPA and Binary file Notebook
* Facebook (fairseq) Manifest Notebook
* Example of Wav2Vec2 with KenLM + Hotwords Pipeline


## Get Average SNR for File of Audio Files
### Description
#### **What is SNR**
Signal-to-Noise Ratio (SNR) in audio processing is the ratio (usually in decibels) between the signal power to noise power of a transmitted signal. It is useful to determining how much power noise has over your audio files. Higher numbers typically mean the useful information (actual signal) is more powerful than the unwanted information (noise). For example, if an audio wave has an SNR of 85 dB that means the audio signal is 85 dB higher than the noise. When analyzing audio files for ASR research it is important to not only take not of the background noise present in your audio files but to also quantify it to know just how much it will impact your systems.

#### **What do I do if my SNR is low?**
If you have a low SNR (the noise is more powerful than the actual signal) there are several options to address the high level of noise power in your audio:
1. Wiener Filter ([for more info]())
2. Butterworth Filter ([for more info](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html))
3. Denoising with Fast Fourier Transform ([for more info](https://www.youtube.com/watch?v=s2K1JfNR7Sc&t=428s))
4. Fine-tune model with data that has been augmented to include noise to make the system more robust
5. Use software such as Audacity, Premiere Pro, GarageBand, etc. (not good for automatically processing large datasets though)


### Instructions
**Necessary Installs:**
- pip install scipy
- pip install argparse
- pip install numpy
- import os

1. Install packages listed above
2. Clone repository or download get_avg_snr.py
3. Open OS command line interface (Command Prompt in Windows)
4. Change directory to directory of get_avg_snr.py
5. Use following command:
```
python get_avg_snr.py --path-to-audio-files C:/your/path/to/your/audio/files/folder
```
*Note, if your files are not .wav files (default) you need to alert the script:*
```
python get_avg_snr.py --path-to-audio-files C:/your/path/to/your/audio/files/folder --audio-file-type mp3
```

*Larger folders might take a bit longer to process*

## Phonetic Noising Script (GEC Usage)

### Description

### Instructions


## JSON Formatter for Huggingface Seq2Seq (GEC Usage)

### Description

### Instructions

## Audio Data Augmentation

### Description
This is an extension of the work done [here](https://github.com/waveletdeboshir/speechaugs/) but scaled to perform data augmentations on a dataset rather than individual examples. 

### Instructions


## KenLM ARPA and Binary File Notebook

### Description

### Instructions


## Facebook (fairseq) Manifest Creation Notebook

### Description

### Instructions


## Example of Wav2Vec2 with KenLM + Hotwords Pipeline

### Description

### Instructions
