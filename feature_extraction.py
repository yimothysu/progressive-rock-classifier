import os
import librosa
import librosa.display
import numpy as np

# import matplotlib.pyplot as plt

from tqdm import tqdm


def feature_extraction(input_folder, output_folder):

    # Choosing which subfolder the features will go to depending on the folder opened

    if input_folder == "data_preprocessing/non_prog_rock":
        subfolder = "non_prog_rock_features"
    else:
        subfolder = "prog_rock_features"

    # Initiating for loop to iterate through every single snippet

    for root, dirs, files in os.walk(input_folder):
        for filename in tqdm(files):

            # Creating path which is then used to load in the snippet through librosa

            path = os.path.join(root, filename)

            y, sr = librosa.load(path)

            # Extracting MFCC feature from snippet

            MFCC = librosa.feature.mfcc(y=y, sr=sr)

            # Creating and saving MFCC feature of snippet in MFCC directory corresponding to prog or non prog features

            dir_MFCC = os.path.join(output_folder, subfolder, "MFCC")
            os.makedirs(dir_MFCC, exist_ok=True)
            np.save(
                os.path.join(dir_MFCC, f"{os.path.splitext(filename)[0]}.npy"), MFCC
            )

            # Commands used for plotting MFCC

            # plt.figure(figsize=(10, 4))
            # librosa.display.specshow(MFCC, x_axis='time')
            # plt.colorbar(format='%+2.0f dB')
            # plt.show()

            # Extracting Mel-spectrogram from snippet and converting into decibels

            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Creating and saving mel-spectrogram of snippet in Mel-Spectrogram directory corresponding to prog or non-prog features

            dir_mel = os.path.join(output_folder, subfolder, "Mel-Spectrogram")
            os.makedirs(dir_mel, exist_ok=True)
            np.save(
                os.path.join(dir_mel, f"{os.path.splitext(filename)[0]}.npy"),
                mel_spec_db,
            )

            # Plotting commands to display Mel-spectrogram

            # plt.figure(figsize=(10, 4))
            # librosa.display.specshow(mel_spec_db, x_axis='time')
            # plt.colorbar(format='%+2.0f dB')
            # plt.show()

            # Extracting chromogram feature from snippet

            chrom = librosa.feature.chroma_stft(y=y, sr=sr)

            # Creating and saving Chromogran of snippet in 'Chromogram' directory corresponding prog or non prog directory

            dir_chrom = os.path.join(output_folder, subfolder, "Chromogram")
            os.makedirs(dir_chrom, exist_ok=True)
            np.save(
                os.path.join(dir_chrom, f"{os.path.splitext(filename)[0]}.npy"), chrom
            )

            # Plotting commands to display chromagram

            # plt.figure(figsize=(10, 4))
            # librosa.display.specshow(chrom, x_axis='time')
            # plt.colorbar(format='%+2.0f dB')
            # plt.show()

            # Extracting beat onset strenght of snippet and normalized

            beat_onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            beat_onset_strength_norm = librosa.util.normalize(beat_onset_strength)

            # Creating and saving beat onset strengths in 'Beat Onset Strength' directory under the corresponding prog or non prog directory

            dir_beat = os.path.join(output_folder, subfolder, "Beat Onset Strength")
            os.makedirs(dir_beat, exist_ok=True)
            np.save(
                os.path.join(dir_beat, f"{os.path.splitext(filename)[0]}.npy"),
                beat_onset_strength_norm,
            )

            # Plotting commands to display beat onset strength

            # plt.figure(figsize=(10, 4))
            # plt.plot(librosa.times_like(beat_onset_strength), beat_onset_strength_norm)
            # plt.show()


input_folders = ["data_preprocessing/non_prog_rock", "data_preprocessing/prog_rock"]
output_folder = "features"

for folder in input_folders:
    feature_extraction(folder, output_folder)
