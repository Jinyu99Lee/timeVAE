import os
import numpy as np
import pickle
import yaml

SCALER_FNAME = "scaler.pkl"


def load_yaml_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    return loaded


def load_data(data_dir: str, dataset: str) -> np.ndarray:
    """
    Load data from a dataset located in a directory.

    Args:
        data_dir (str): The directory where the dataset is located.
        dataset (str): The name of the dataset file (without the .npz extension).

    Returns:
        np.ndarray: The loaded dataset.
    """
    return get_npz_data(os.path.join(data_dir, f"{dataset}.npz"))


def save_data(data: np.ndarray, output_file: str) -> None:
    """
    Save data to a .npz file.

    Args:
        data (np.ndarray): The data to save.
        output_file (str): The path to the .npz file to save the data to.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(output_file, data=data)


def get_npz_data(input_file: str) -> np.ndarray:
    """
    Load data from a .npz file.

    Args:
        input_file (str): The path to the .npz file.

    Returns:
        np.ndarray: The data array extracted from the .npz file.
    """
    loaded = np.load(input_file)
    return loaded["data"]


def _full_train_recent_blocks_valid_data(data: np.ndarray) -> np.ndarray:
    """
    Build a validation set by copying three recent-year sample blocks.

    The slices are expressed on the sample axis and intentionally match the
    experiment protocol:
    - last year's last 122 samples: data[-122:]
    - second-last year's middle 122 samples: data[-365-244:-365-122]
    - third-last year's earliest 122 samples: data[-730-366:-730-244]
    """
    valid_slices = (
        slice(-122, None),
        slice(-365 - 244, -365 - 122),
        slice(-730 - 366, -730 - 244),
    )
    if data.shape[0] < 1096:
        raise ValueError(
            "split_method='full_train_recent_blocks' requires at least 1096 "
            f"samples, got {data.shape[0]}."
        )

    valid_parts = [data[valid_slice].copy() for valid_slice in valid_slices]
    expected_block_size = 122
    for idx, valid_part in enumerate(valid_parts):
        if valid_part.shape[0] != expected_block_size:
            raise ValueError(
                "Recent-block validation split produced an unexpected block size "
                f"for block {idx}: expected {expected_block_size}, got "
                f"{valid_part.shape[0]}."
            )

    return np.concatenate(valid_parts, axis=0)


def _shuffle_samples(data: np.ndarray, seed: int) -> np.ndarray:
    shuffled = data.copy()
    np.random.seed(seed)
    np.random.shuffle(shuffled)
    return shuffled


def split_data(
    data: np.ndarray,
    valid_perc: float,
    shuffle: bool = True,
    seed: int = 123,
    split_method: str = "tail_holdout",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the data into training and validation sets.

    Args:
        data (np.ndarray): The dataset to split.
        valid_perc (float): The percentage of data to use for validation for
                            split_method="tail_holdout".
        shuffle (bool, optional): Whether to shuffle the returned training data.
                                  Validation data is never shuffled.
                                  Defaults to True.
        seed (int, optional): The random seed to use for shuffling training data.
                              Defaults to 123.
        split_method (str, optional): Split strategy. "tail_holdout" reserves
                                      the last valid_perc samples for validation,
                                      then shuffles only the training split.
                                      "full_train_recent_blocks" uses all samples
                                      for shuffled training and copies validation
                                      from three recent-year blocks.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the training data and
                                       validation data arrays.
    """
    if split_method == "full_train_recent_blocks":
        train_data = _shuffle_samples(data, seed) if shuffle else data.copy()
        valid_data = _full_train_recent_blocks_valid_data(data)
        return train_data, valid_data
    if split_method != "tail_holdout":
        raise ValueError(
            f"Unknown split_method={split_method!r}. Expected 'tail_holdout' "
            "or 'full_train_recent_blocks'."
        )

    N = data.shape[0]
    N_train = int(N * (1 - valid_perc))
    train_data = data[:N_train].copy()
    valid_data = data[N_train:].copy()

    if shuffle:
        train_data = _shuffle_samples(train_data, seed)

    return train_data, valid_data


class MinMaxScaler:
    """Min Max normalizer.
    Args:
    - data: original data

    Returns:
    - norm_data: normalized data
    """

    def fit_transform(self, data):
        self.fit(data)
        scaled_data = self.transform(data)
        return scaled_data

    def fit(self, data):
        self.mini = np.min(data, 0)
        self.range = np.max(data, 0) - self.mini
        return self

    def transform(self, data):
        numerator = data - self.mini
        scaled_data = numerator / (self.range + 1e-7)
        return scaled_data

    def inverse_transform(self, data):
        data *= self.range
        data += self.mini
        return data


def inverse_transform_data(data, scaler):
    return scaler.inverse_transform(data.copy())


def scale_data(train_data, valid_data):
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    scaled_valid_data = scaler.transform(valid_data)
    return scaled_train_data, scaled_valid_data, scaler


def save_scaler(scaler: MinMaxScaler, dir_path: str) -> None:
    """
    Save a MinMaxScaler to a file.

    Args:
        scaler (MinMaxScaler): The scaler to save.
        dir_path (str): The path to the directory where the scaler will be saved.

    Returns:
        None
    """
    os.makedirs(dir_path, exist_ok=True)
    scaler_fpath = os.path.join(dir_path, SCALER_FNAME)
    with open(scaler_fpath, "wb") as file:
        pickle.dump(scaler, file)


def load_scaler(dir_path: str) -> MinMaxScaler:
    """
    Load a MinMaxScaler from a file.

    Args:
        dir_path (str): The path to the file from which the scaler will be loaded.

    Returns:
        MinMaxScaler: The loaded scaler.
    """
    scaler_fpath = os.path.join(dir_path, SCALER_FNAME)
    with open(scaler_fpath, "rb") as file:
        scaler = pickle.load(file)
    return scaler
