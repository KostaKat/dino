import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Optional
import torch
from min_pre import MineralPreprocessing  # Import pre-processing class
class MineralDataset(Dataset):
    """
    A custom PyTorch Dataset for loading mineral image patches and their corresponding metadata.

    This dataset class handles the retrieval and preprocessing of image patches based on metadata provided in a CSV file.
    It supports optional transformations and preprocessing steps, making it suitable for training deep learning models.

    Attributes:
        patches_dir (str): Directory where patch images are stored.
        transform (callable, optional): Optional transform to be applied on an image.
        preprocessing (MineralPreprocessing, optional): Optional preprocessing to be applied on an image.
        metadata (pd.DataFrame): DataFrame containing metadata loaded from the CSV file.
        image_name_to_id (Dict[str, int]): Mapping from original image names to unique integer IDs.
    """

    def __init__(
        self,
        csv_path: str,
        patches_dir: str,
        transform: Optional[callable] = None,
        preprocessing: Optional[MineralPreprocessing] = None
    ) -> None:
        """
        Initializes the MineralDataset with metadata, patch directory, and optional transformations.

        Args:
            csv_path (str): Path to the CSV file containing patch metadata.
            patches_dir (str): Directory where patch images are stored.
            transform (callable, optional): Optional transform to be applied on an image.
            preprocessing (MineralPreprocessing, optional): Optional preprocessing to be applied on an image.

        Raises:
            FileNotFoundError: If the CSV file does not exist at the specified path.
            ValueError: If the required columns are missing in the CSV file.
        """
        self.patches_dir = patches_dir
        self.transform = transform
        self.preprocessing = preprocessing

        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"The CSV file does not exist: {csv_path}")

        # Load the CSV metadata into a DataFrame
        self.metadata = pd.read_csv(csv_path)

        required_columns = {'patch_name', 'original_image', 'x_coordinate', 'y_coordinate'}
        if not required_columns.issubset(self.metadata.columns):
            missing = required_columns - set(self.metadata.columns)
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # Create a mapping from original image names to unique IDs
        self.image_name_to_id = {
            name: idx for idx, name in enumerate(self.metadata['original_image'].unique())
        }

    def __len__(self) -> int:
        """
        Returns the total number of patches in the dataset.

        Returns:
            int: Number of patches.
        """
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Retrieves the image patch and its associated metadata at the specified index.

        Args:
            idx (int): Index of the patch to retrieve.

        Raises:
            IndexError: If the index is out of bounds.
            FileNotFoundError: If the image file does not exist.
            ValueError: If image loading or preprocessing fails.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing:
                - The processed image tensor.
                - The image ID (or any other label or identifier you want to return).
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}.")

        # Retrieve the row corresponding to the given index
        row = self.metadata.iloc[idx]
        patch_name = row['patch_name']
        original_name = row['original_image']
        x_pos = row['x_coordinate']
        y_pos = row['y_coordinate']

        # Construct the full path to the patch image
        image_path = os.path.join(self.patches_dir, patch_name)

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Load the image and convert to RGB
            with Image.open(image_path) as img:
                image = img.convert('RGB')  # Ensure 3 channels
            image = np.array(image)

            # Apply preprocessing if provided
            if self.preprocessing:
                image = self.preprocessing.preprocess_image(image)
            image = Image.fromarray(image)
            # Get the unique image ID (or you can return any other relevant metadata)
            image_id = self.image_name_to_id[original_name]

            # Apply transformation if provided
            if self.transform:
                image = self.transform(image)
            

        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {e}")
        
        return image, image_id  # Returning both the image and its corresponding ID

