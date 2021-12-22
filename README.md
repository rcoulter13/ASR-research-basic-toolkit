# ASR-research-basic-toolkit
Python scripts for basic Automatic Speech Recognition tasks such as finding the SNR, noising data phonetically for Grammar Error Checking (seq2seq) training, easy data augmentation, easy KenLM creation, etc.

These tools are all things I find useful while doing any kind of Automatic Speech Recognition (ASR) research. These are things I wish had existed when I was first learning about ASR and so I thought I'd share them to save other people on the same path some time :) May they serve you well! 

### *Contents*
* Get Average SNR for File of Audio Files
* Phonetic Noising Script for (GEC usage)
* JSON Formatter for Huggingface Seq2Seq (GEC usage)
* Audio Data Augmentation
* KenLM ARPA and Binary file Notebook
* Facebook (fairseq) Make Manifest Notebook
* Example of Wav2Vec2 with KenLM + Hotwords Pipeline (using PyCTCDecode)


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

**Steps:**
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
This script is useful for creating your own Grammar Error Checker using a sequence to sequence model. The basic idea is to take a sequence to sequence model (usually pre-trained on a language translation task) and fine-tune it with the ungrammatical sentences being the input 'language' and the grammatical sentences being the output 'language'. For modern ASR tasks the grammatical errors are often phonetically-based. For example, there are instances where 'k' is guessed instead of 'g' or 'p' instead of 'b'. With this pattern in mind, I developed a script that noises data but rather than noising it with the regular insertions, deletions, swaps, etc., it does phonetic-based noising including assimilation, homophone swapping and manner of articulation swapping.

For example, if you want 10% of your sentence phonetically noised, 10% of the words in each sentence will randomly receive any of the following transformations:

**Homophone**

*Swap one word with another word that is spelled differently but sounds the same*

sea -> see

**Assimilation**

*Assimilation occurs when speakers talk too fast and (character by character) ASR systems blur words together, neglecting proper spacing*

for real -> foreal

**Manner of Articulation**

*From phonetics, this method swaps one sound (or letter) with one from the category of similar manner of articulation sounds. For example, a 'k' is a plosive and can be replaced with other plosives such as 'g', 't', 'p'. As unusual as this sounds, this method was based off naturall occuring behaviors in Wav2Vec2 out-of-the-box transcripts so I mimicked it when attempting to establish a functional GEC. This method also accounts for deletions as each swap also has the chance of swapping with a blank character instead of a similar manner of articulation*

cheese -> cheeze


*Note that the types of phonetic noising are weighted based on their relative impact on the sentence they modify so some types will be used more readily than others*

### Instructions
**Necessary Installs:**
- pip install wordhoard
- pip install tqdm
- pip install nltk
- pip install argparse

**Steps:**
1. Install packages listed above
2. Clone repository or download phoneticNoiser.py
3. Open OS command line interface (Command Prompt in Windows)
4. Change directory to directory of phoneticNoiser.py
5. There are four arguments to the script:
    a. --path - path to text file containing transcripts for phonetic noising
    b. --percent - percent of data to be noised per sentence
    c. --outpath - path to which to write the noised/unnoised sentence pair csv
    d. --output-name - name for new noised csv file
   Make sure to have these values handy.
7. Take those values and use the following command with your values:
```
python phoneticNoiser.py --path C:\your\path\to\transcript\txt\file --percent 0.12 --outpath C:\your\path\to\output\folder --output-name noisedSents_demo
```


## JSON Formatter for Huggingface Seq2Seq (GEC Usage)
### Description
Reads in a csv file that contains 'ungrammatical' and 'grammatical' sentences for the sequence to sequence Grammar Error Checker task mentioned above and reformats these target and reference predictions into the following JSON structure:

```
{
translation:
[{"befr" : "he eight the cheeze sandwich", "en" : "he ate the cheese sandwich"}
{"befr" : "she kan't sea the kat", "en" : "she can't see the cat"}]
}
```
This allows a transformer model or other sequence to sequence architecture to read in the dataset and then be fine-tuned as a Grammar Error Checker (GEC) for the target domain.
**It also serves as a train\validation\test splitter**

### Instructions
**Necessary Installs:**
- pip install argparse

**Steps:**
1. Install packages listed above
2. Clone repository or download seq2seq_json_formatter.py
3. Open OS command line interface (Command Prompt in Windows)
4. Change directory to directory of seq2seq_json_formatter.py
5. There are four arguments to the script:
    a. --name - Name to uniquely identify the output JSON file (script will add -train, -test, -val to distinguish types)
    b. --files - Comma separated string of csv files (or paths to files) to process and reformat
    c. --output-dir - Directory to which to save new JSON files
    d. --val-split - Percentage (as decimal) of data to be used for validation
    e. --csvIndices - Column indices from csv to grab for source-target sentences (ex: 5,6 with 5 being source and 6 being target) 
    f. --test-split - Percentage (as decimal) of data to be used for testing [OPTIONAL]
7. Take those values and use the following command with your values:
```
python seq2seq_json_formatter.py --name my-json-file-demo --files C:\your\path\to\csv\file1,C:\your\path\to\csv\file2 --output-dir C:\your\path\to\output\folder --val-split 0.20 --csvIndices 0,1
```


## Audio Data Augmentation
### Description
This is an extension of the work done [here](https://github.com/waveletdeboshir/speechaugs/) but scaled to perform data augmentations on a dataset rather than individual examples. Augmentation types include *Time Stretch*, *Forward Time Stretch*, *Pitch Shift*, *Vocal Tract Length Perturbation*,  *Short Noise Injection*, and *Amplitude Shift*.

*  **Time Stretch** - shifts audio data back in time.
*  **Forward Time Stretch** - shifts audio data forward in time.
*  **Pitch Shift** - shift pitch by *n_steps* semitones.
*  **Vocal Tract Length Perturbation** - similar to pitch shift. VTLP finds the change in formant frequence correlated to the change in vocal tract length.
*  **Short Noise Injection** - Add multiple short bursts of noise (same color) to random points of the waveform.
*  **Amplitude Shift** - Alter amplitude of waveform.

### Instructions
**Necessary Installs:**

*Already included*

**Steps:**
1. Upload audio-data-augment.ipynb to your Google Drive
2. Follow instructions within Colab notebook

## KenLM ARPA and Binary File Notebook
### Description
Clear and easy notebook to create Ken Language Model (KenLM) ARPA and binary files from text files. All you need is a basic .txt file containing the text you want to base your KenLM off of - the most common demonstration is creating a KenLM from a text file of the Bible. 

### Instructions
**Necessary Installs:**

*Already included*

**Steps:**
1. Upload kenlm_notebook.ipynb to your Google Drive
2. Follow instructions within Colab notebook

## Facebook (fairseq) Make Manifest Notebook
### Description
*Note: this is pretty much only if you want to fine-tune a Wav2Vec2 model using Fairseq scripts*

If you want to finetune Wav2Vec2 (at least as of 2021) from Fairseq using a command line interface rather than the Huggingface implementation, it is necessary to create a manifest file and follow the instructions given [here](https://bleepcoder.com/fairseq/680709778/i-want-to-finetune-this-model-with-my-own-audio-files-how). The manifest file stores information the Fairseq training code requies to fine-tune a Wav2Vec2 model including: train.tsv, train.wrd, train.ltr, valid.tsv (or dev_other.tsv depending on the name scheme), valid.wrd (or dev_other.wrd), valid.ltr (or dev_other.ltr), dict.ltr.txt, and lexicon.txt. The code in the notebook along with the modified labels code (modLibri_labels.py, wav2vec2_lexicon.py) included here are intended to make this process easier and to help you avoid all the problems I encountered the first time I attempted this task.

### Instructions
**Necessary Installs:**

*Already included for Notebook*

*For extra scripts (if you haven't already within Colab)*
- pip install argparse
- pip install pandas

**Steps:**
1. Upload make_manifest_notebook.ipynb to your Google Drive
2. Upload modLibri_labels.py, wav2vec2_lexicon.py, and ltr_counter.py also to your Google Drive
3. Follow instructions within Colab notebook

## Example of Wav2Vec2 with KenLM + Hotwords Pipeline (from PyCTCDecode)
### Description
*This is assuming you already have a trained/fine-tuned Wav2Vec2 model and saved your processor to the same directory.*
This notebook offers a completed Wav2Vec2 Inferencing pipeline with Word Error Rate averaging (and individual scoring) as well as results with no language model, with a language model and with a language model plus hotwords for comparison purposes. All the credit for the actual PyCTCDecode code and KenLM code goes to their original authors - the purpose of this notebook is to have the pipeline with all essential code in one place, end-to-end for easy use.

*UPDATE (12-22-21): This pipeline is now nearly identical to the Huggingface Transformers Wav2Vec2 + PyCTCDecode found [here](https://github.com/patrickvonplaten/Wav2Vec2_PyCTCDecode). The main difference is that this version walks you through every step of the process while the latest version from Huggingface only requires an additional argument to a command line implementation. If you want to see what is going on 'under the hood' then I would recommend the Colab notebook included here; otherwise, I would recommend that you follow the instructions given [here](https://github.com/patrickvonplaten/Wav2Vec2_PyCTCDecode)

### Instructions
**Necessary Installs:**

*Already included*

**Steps:**
1. Upload WER_PyCTCDecode_Notebook.ipynb to your Google Drive
2. Follow instructions within Colab notebook
