import os
from mutagen.mp3 import MP3


def get_cumulative_length(directory):
    total_length = 0
    total_count = 0

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".mp3"):
                filepath = os.path.join(root, filename)
                audio = MP3(filepath)
                duration = audio.info.length
                total_length += duration
                total_count += 1

    return total_length, total_count


# Specify the directory path
directory_path = "data/CAP6610SP24_test_set"

# Get the cumulative length in seconds
cumulative_length, count = get_cumulative_length(directory_path)

# Convert seconds to hours, minutes, and seconds
hours = int(cumulative_length // 3600)
minutes = int((cumulative_length % 3600) // 60)
seconds = int(cumulative_length % 60)

print(f"Cumulative runtime length: {hours:02d}:{minutes:02d}:{seconds:02d}")
print(f"Total number of files: {count}")
