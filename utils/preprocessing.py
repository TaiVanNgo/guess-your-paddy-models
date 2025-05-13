# Data manipulation libraries
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from collections import defaultdict

# Import custom libraries
import utils.visualizations as ut


def apply_img_cutout(image, percentage=0.2, rng=None):
    """
    Cut the image based on the percentage

    - image: the image to cut
    - percentage: the percentage of cutout
    - rng: Random number generator for reproducibility.
    """

    h, w = image.shape[:2]
    cutout_size_h = int(h * percentage)
    cutout_size_w = int(w * percentage)

    # Random position for cutout
    x = rng.randint(0, w - cutout_size_w)
    y = rng.randint(0, h - cutout_size_h)

    # Create copy of the image
    img_cutout = image.copy()
    img_cutout[y : y + cutout_size_h, x : x + cutout_size_w] = 0

    return img_cutout


def zoom_in_crop(image, zoom_factor=1.2):
    """
    Zoom in on the image by cropping the edges and resizing back to original size

    - image: the image file
    - zoom_factor: the zoom ratio
    """
    h, w = image.shape[:2]

    # Calculate crop dimensions
    crop_h = int(h / zoom_factor)
    crop_w = int(w / zoom_factor)

    # Calculate offsets to center the crop
    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2

    # Crop and resize
    cropped = image[start_y : start_y + crop_h, start_x : start_x + crop_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return zoomed


def random_crop_resize(image, crop_percent=0.8, rng=None):
    """
    Random crop of the image followed by resize to original dimensions

    - image: input image
    - crop_percent: Percentage of the image to keep
    - rng: Random number generator for reproducibility.
    """
    if rng is None:
        rng = np.random

    h, w = image.shape[:2]
    crop_h = int(h * crop_percent)
    crop_w = int(w * crop_percent)

    # Random starting point for crop
    start_y = rng.randint(0, h - crop_h + 1)
    start_x = rng.randint(0, w - crop_w + 1)

    # Crop and resize
    cropped = image[start_y : start_y + crop_h, start_x : start_x + crop_w]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return resized


def rotate_image(image, angle):
    """
    Rotates an image by the specified angle.

    Parameters:
    - image: Input image
    - angle: Array of angles in degrees

    Returns:
    - Rotated image with the same dimensions as input
    """
    # Get image dimensions and center
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation while maintaining original dimensions
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,  # Avoid black borders
    )

    return rotated


def df_augmentation(
    df,
    output_csv_path="./inputs/meta_train_augmented.csv",
    img_augmented_dir="./inputs/train_augmented_images",
    random_state=42,
):
    """
    Augments underrepresented paddy varieties using various image transformations.

    Parameters:
    - df: Original dataFrame containing image metadata
    - output_csv_path: Path to save the augmented metadata CSV
    - img_original_dir: Directory containing original images
    - img_augmented_dir: Directory to save augmented images
    - random_state: Seed for reproducible augmentations

    Returns:
    - augmented_df: DataFrame containing original and augmented image metadata
    - augmented_count: Dictionary counting new images per variety
    """
    # Fixed seed RNG for consistent augmentations
    rng = np.random.RandomState(random_state)

    # ensure that the img_augmented_dir exists
    os.makedirs(img_augmented_dir, exist_ok=True)

    target_varieties = df["variety"].value_counts()

    # Create new rows for augmented images
    new_rows = []
    augmented_count = defaultdict(int)

    # Define augmentation strategies based on variety count
    VERY_LOW_COUNT = 50
    LOW_COUNT = 120
    MID_LOW_COUNT = 400
    MID_COUNT = 1000

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating augmentations"):
        image_id, label, variety, age = (
            row["image_id"],
            row["label"],
            row["variety"],
            row["age"],
        )

        # Get image path
        image_path = row["image_path"]

        variety_count = target_varieties[variety]

        # if the current variety count exceeds the MID_COUNT (1000) already -> skip
        if variety_count > MID_COUNT:
            continue

        # if the path doesn't exist
        if not os.path.exists(image_path):
            print(f"[ERROR] Invalid image path: {image_path}")
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"[ERROR] Failed to load image: {image_path}")
            continue

        augmentations = []

        # For very low count (<=50)
        if variety_count < VERY_LOW_COUNT:
            # Rotations augmentations
            for angle in [-15, -10, -5, 5, 10, 15]:
                rotated = rotate_image(image, angle)
                augmentations.append((f"rot_{angle}", rotated))

            # Horizontal flip
            augmentations.append(("hflip", cv2.flip(image, 1)))

            # Vertical flip
            augmentations.append(("vflip", cv2.flip(image, 0)))

            # Brightness adjustment positive_20 and negative_20(+20% and -20%)
            augmentations.append(
                ("bright_p20", cv2.convertScaleAbs(image, alpha=1.2, beta=0))
            )
            augmentations.append(
                ("bright_n20", cv2.convertScaleAbs(image, alpha=0.8, beta=0))
            )

            # Zoom-in crop
            augmentations.append(("zoom", zoom_in_crop(image)))

            # Cutout
            augmentations.append(("cutout", apply_img_cutout(image, rng=rng)))

        # Low count variety (< 100)
        elif variety_count < LOW_COUNT:
            # Apply rotations
            for angle in [-10, -5, 5, 10]:
                rotated = rotate_image(image, angle)
                augmentations.append((f"rot_{angle}", rotated))

            # Brightness adjustments (+20% and -20%)
            augmentations.append(
                ("bright_p20", cv2.convertScaleAbs(image, alpha=1.2, beta=0))
            )
            augmentations.append(
                ("bright_n20", cv2.convertScaleAbs(image, alpha=0.8, beta=0))
            )

        # Lower than 400
        elif variety_count < MID_LOW_COUNT:
            # Limited rotations
            for angle in [-5, 5]:
                rotated = rotate_image(image, angle)
                augmentations.append((f"rot_{angle}", rotated))

            # Random crop and resize
            augmentations.append(("crop_resize", random_crop_resize(image, rng=rng)))

        # Lower than 800 counts
        elif variety_count < MID_COUNT:
            for angle in [-5, 5]:
                rotated = rotate_image(image, angle)
                augmentations.append((f"rot_{angle}", rotated))

        # Apply all selected augmentations
        for aug_name, aug_img in augmentations:
            # Extract base name without extension
            base_name = os.path.splitext(image_id)[0]

            new_image_id = f"{base_name}_{aug_name}.jpg"

            label_dir = os.path.join(img_augmented_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            new_path = os.path.join(label_dir, f"{new_image_id}")

            # If the image file doesn't exist
            if not os.path.exists(new_path):
                cv2.imwrite(new_path, aug_img)

            new_rows.append(
                {
                    "image_id": new_image_id,
                    "label": label,
                    "variety": variety,
                    "age": age,
                    "image_path": new_path,
                }
            )
            augmented_count[variety] += 1

    # Create augmented DataFrame
    augmented_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    # Save augmented metadata
    augmented_df.to_csv(output_csv_path, index=False)

    ut.print_header("Augmentation Summary")
    print(f"Original dataset: {len(df)} images")
    print(f"Augmented dataset: {len(augmented_df)} images")
    print(f"Added {len(new_rows)} augmented images")

    return augmented_df, augmented_count

