import os
import random
from tqdm import tqdm
import librosa
import soundfile as sf

def collect_song_paths(input_folders):
    """Collect all song paths from the input directories."""
    all_song_paths = []
    for folder in input_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                all_song_paths.append(file_path)
    return all_song_paths

def split_songs(song_paths, split_ratio=0.8):
    """Split song paths into training and validation sets."""
    random.seed(42)  # Ensure reproducibility
    random.shuffle(song_paths)
    split_point = int(len(song_paths) * split_ratio)
    return song_paths[:split_point], song_paths[split_point:]

def process_song(song_path, output_folder, sr=11025, set_type='train'):
    # Extract information for path construction
    root, file = os.path.split(song_path)
    if 'Not_Progressive_Rock' in root:
        subfolder_name = 'test_non_prog_rock'
    elif 'Progressive Rock Songs' in root:
        subfolder_name = 'test_prog_rock'
    elif 'other' in root:
        subfolder_name = 'test_other'
    else:
        subfolder_name = 'unknown_category'  # Handles unexpected categories
    
    song_name = os.path.splitext(file)[0]
    
    # Construct the output path
    output_path = os.path.join(output_folder, subfolder_name, song_name)
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Load, process, and split the audio file into chunks
        audio, _ = librosa.load(song_path, sr=sr, mono=True)
        audio_trimmed, _ = librosa.effects.trim(audio)
        chunks = [audio_trimmed[i:i + sr * 10] for i in range(0, len(audio_trimmed), sr * 10)]
        
        # Save the chunks
        for idx, chunk in enumerate(chunks):
            chunk_path = os.path.join(output_path, f"{idx+1}.wav")
            sf.write(chunk_path, chunk, sr)
    except Exception as e:
        print(f"Error processing {file}: {e}")

def main(input_folder, output_folder, sr=22050):
    # Define specific subfolders
    input_folders = [os.path.join(input_folder, "Not_Progressive_Rock"), 
                     os.path.join(input_folder, "Progressive Rock Songs"),
                     os.path.join(input_folder, "other")]  # Added 'other' folder
    
    # Collect all songs and split them
    song_paths = collect_song_paths(input_folders)
    train_songs, valid_songs = split_songs(song_paths)
    
    # Process songs for each set
    for song_path in tqdm(train_songs, desc="Processing Training Songs"):
        process_song(song_path, output_folder, sr, 'train')
    for song_path in tqdm(valid_songs, desc="Processing Validation Songs"):
        process_song(song_path, output_folder, sr, 'valid')

# Example usage
input_folder = "data/CAP6610SP24_test_set"
output_folder = "processed_data"
main(input_folder, output_folder)
