"""
Author: Riah Coulter
Date: October 12, 2021 (Updated December 21, 2021)
Purpose: Computes average signal-to-noise ratio for given folder of audio files
Parameters:
    path-to-audio-files - path to individual audio files from which average SNR is calculated
"""
import scipy.io.wavfile as wavfile
import argparse
#from scipy import stats -> deprecated
import numpy as np
import os

def get_snr(file):
    """
    Takes an audio file, opens it, calculates the SNR for the individual
    audio file then returns the SNR.
    
    :params: [str] path to specific audio file
    :returns: [float] SNR value
    """
    if (os.path.isfile(file)):
        axis = 0
        ddof = 0
        a = wavfile.read(file)[1]
        mx = np.amax(a)
        a  = np.divide(a,mx)
        a  = np.square(a)
        a  = np.asanyarray(a)
        m  = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m/sd)
    else:
        print(f"{file} not found.")
        return 0


def get_average_snr(folder_dir: str, file_type: str):
    """
    Iterates through all audio files that end with given
    file type (default=wav), gets the SNR for each audio
    file then prints the average SNR over all audio files.

    :params: [str] folder_dir - path to file containing target
    audio files, [str] file_type - type of audio file to search
    for (wav, mp3, etc.)
    :returns: Nothing. Prints average SNR over audio files.
    """
    total = 0
    count = 0
    for root, _, files in os.walk(folder_dir):
        for name in files:
            if name.endswith(f'.{file_type}'):
                path = os.path.join(root, name)
                snr  = get_snr(path)
                total += snr
                count += 1
    try:
        print(f"\nFolder: {folder_dir}")
        print(f"\nAverage SNR:\n{total/count}dB")
    except ZeroDivisionError:
        print('ZeroDivisionError: Files not processed correctly. Check file format.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-audio-files", required=True, help='Path to individual audio files (MUST BE .wav) to be processed.')
    parser.add_argument("--audio-file-type", required=False, default='wav', help='Type of audio file. Examples include: wav, mp3, etc.')
    args = parser.parse_args()
    print('\nFILE TYPE:', args.audio_file_type)
    get_average_snr(args.path_to_audio_files, args.audio_file_type)

if __name__ == "__main__":
    main()