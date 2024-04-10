import os

def count_songs(base_path):
    """
    Count the number of songs in each set (train and valid) within the base directory.

    Args:
        base_path (str): The base directory containing the 'train' and 'valid' subdirectories.

    Returns:
        dict: A dictionary with the counts of songs for training and validation sets.
    """
    counts = {'train': 0, 'valid': 0}
    for set_type in ['train', 'valid']:
        set_path = os.path.join(base_path, set_type)
        if not os.path.exists(set_path):
            print(f"The path {set_path} does not exist.")
            continue

        for genre in os.listdir(set_path):
            genre_path = os.path.join(set_path, genre)
            if os.path.isdir(genre_path):
                counts[set_type] += len([name for name in os.listdir(genre_path) if os.path.isdir(os.path.join(genre_path, name))])

    return counts

# Example usage
base_path = "processed_data"  # Adjust the base path as necessary
counts = count_songs(base_path)
print(f"Number of songs in the training set: {counts['train']}")
print(f"Number of songs in the validation set: {counts['valid']}")
