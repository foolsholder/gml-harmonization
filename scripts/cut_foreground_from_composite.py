import argparse
import cv2
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from typing import List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset.')

    return parser.parse_args()


def cut_foregrounds(dataset_path: Path) -> None:
    composite_folder: Path = dataset_path / 'composite_images'
    mask_folder: Path = dataset_path / 'masks'

    target_dir: Path = dataset_path / 'foreground_objects'
    target_dir.mkdir(exist_ok=True)

    composite_images_paths: List[Path] = list(composite_folder.rglob('*.jpg'))
    # list of Path's objects sorted lexicographically by strings
    composite_images_paths.sort()

    count_cimages: int = len(composite_images_paths)

    prev_mask_name: str = ''
    mask = None

    for image_path in tqdm(composite_images_paths, total=count_cimages):
        composite_image = cv2.imread(str(image_path))

        image_name: str = image_path.stem

        separ_idx: int = image_name.rfind('_')
        mask_name: str = image_name[:separ_idx]

        mask_path: Path = mask_folder / (mask_name + '.png')

        if mask_name != prev_mask_name:
            mask = cv2.imread(str(mask_path))
            mask = mask[:, :, :1].astype(np.float32)# / 255.
            prev_mask_name = mask_name

        foreground_object = composite_image.astype(np.float32) * mask
        foreground_object = foreground_object.astype(np.uint8)

        target_path: Path = target_dir / image_path.name
        cv2.imwrite(str(target_path), foreground_object)


if __name__ == '__main__':
    args = parse_args()
    cut_foregrounds(Path(args.dataset_path))
