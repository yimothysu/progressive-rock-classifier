import os
import random

from tqdm import tqdm
import librosa
import soundfile as sf

def process_audio_files(input_folder, output_folder, sr=11025):
    # Fix the random seed
    random.seed(42)
    
    for root, dirs, files in os.walk(input_folder):
        for file in tqdm(files):
            try:
                # Decide whether the file goes to training or validation set
                set_type = "train" if random.random() < 0.8 else "valid"
                
                # Construct the path to the current file
                path = os.path.join(root, file)
                
                # Load the audio file
                audio, sr_orig = librosa.load(path, sr=sr)  # Load with the target sample rate

                # Trim silence from the start and end
                audio_trimmed, _ = librosa.effects.trim(audio)

                # Resample if necessary
                if sr_orig != sr:
                    audio_resampled = librosa.resample(audio_trimmed, orig_sr=sr_orig, target_sr=sr)
                else:
                    audio_resampled = audio_trimmed
                
                # Determine the genre based subfolder
                subfolder_name = "non_prog_rock" if "Top_Of_The_Pops" in root or "Other_Songs" in root else "prog_rock" if "Progressive_Rock_Song" in root else None
                if subfolder_name is None:
                    continue  # Skip unknown directories

                # Adjust the path for set type (train or valid) and genre subfolder
                set_type_genre_path = os.path.join(output_folder, set_type, subfolder_name)

                # Ensure the directory exists
                os.makedirs(set_type_genre_path, exist_ok=True)

                # Create a unique folder for each song within its genre and set type
                song_folder_name = os.path.splitext(file)[0]
                song_folder_path = os.path.join(set_type_genre_path, song_folder_name)
                os.makedirs(song_folder_path, exist_ok=True)

                # Split into 10-second chunks
                chunk_length = 10 * sr
                chunks = [
                    audio_resampled[i:i+chunk_length] for i in range(0, len(audio_resampled), chunk_length)
                ]

                # Save each chunk
                for idx, chunk in enumerate(chunks):
                    output_file_name = f"{idx+1}.wav"  # Name chunks numerically
                    output_path = os.path.join(song_folder_path, output_file_name)
                    sf.write(output_path, chunk, sr)

            except Exception as e:
                print(f"Error processing {file}: {e}")

# Example usage
input_folders = ["data/Not_Progressive_Rock", "data/Progressive_Rock_Songs"]
output_folder = "processed_data"
for folder in input_folders:
    process_audio_files(folder, output_folder)