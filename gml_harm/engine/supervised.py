import torch
from albumentations import Compose
from catalyst import dl
from typing import Any, Mapping

from ..data.transforms import ToOriginalScale


class SupervisedTrainer(dl.Runner):
    def get_engine(self) -> dl.IEngine:
        return dl.DeviceEngine('cuda:0')

    def __init__(self, restore_scale=True):
        super(SupervisedTrainer, self).__init__()
        self.to_original_scale = None
        if restore_scale is not None:
            self.to_original_scale = ToOriginalScale()

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        images = batch['images']
        masks = batch['masks']
        targets = batch['targets']

        model = self.model['model']
        outputs = model(images, masks)

        self.batch = {
            'outputs': outputs,
            'targets': targets,
            'masks': masks,
            'targets_and_masks': (targets, masks)
        }

        if self.to_original_scale is not None:
            outputs_255 = self.to_original_scale.apply(outputs)
            outputs_255 = torch.clip(outputs_255, 0, 255.)
            targets_255 = self.to_original_scale.apply(targets)
            self.batch.update({
                'outputs_255': outputs_255,
                'targets_255': targets_255,
                'targets_and_masks_255': (targets_255, masks)
            })