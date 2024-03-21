import os
import librosa
import soundfile as sf

def process_audio_files(input_folder, output_folder, sr=11025):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            try:
                # Construct the path to the current file
                path = os.path.join(root, file)
                # Load the audio file
                audio, sr_orig = librosa.load(path, sr=11025)  # Load with original sample rate

                # Trim silence from the start and end
                audio_trimmed, _ = librosa.effects.trim(audio)

                # Resample to 11025 Hz if necessary
                if sr_orig != sr:
                    audio_resampled = librosa.resample(audio_trimmed, orig_sr=sr_orig, target_sr=sr)
                else:
                    audio_resampled = audio_trimmed

                # Determine the output subfolder based on the input path
                if 'Top_Of_The_Pops' in root or 'Other_Songs' in root:
                    subfolder_name = 'non_prog_rock'
                elif 'Progressive_Rock_Song' in root:
                    subfolder_name = 'prog_rock'
                else:
                    continue  # Skip unknown directories

                output_subfolder = os.path.join(output_folder, subfolder_name)

                # Ensure the output subfolder exists
                os.makedirs(output_subfolder, exist_ok=True)

                # Split into 10-second chunks
                chunk_length = 10 * sr  # 10 seconds at the target sample rate
                chunks = [audio_resampled[i:i+chunk_length] for i in range(0, len(audio_resampled), chunk_length)]

                # Save each chunk
                for idx, chunk in enumerate(chunks):
                    output_file_name = f"{os.path.splitext(file)[0]}_chunk{idx}.wav"  # Change to .mp3 if necessary
                    output_path = os.path.join(output_subfolder, output_file_name)
                    sf.write(output_path, chunk, sr)

            except Exception as e:
                print(f"Error processing {file}: {e}")

input_folders = ["data/Progressive_Rock_Songs"]
#input_folders = ["data/Not_Progressive_Rock", "data/Progressive_Rock_Songs"]
output_folder = "data_preprocessing"

for folder in input_folders:
    process_audio_files(folder, output_folder)
