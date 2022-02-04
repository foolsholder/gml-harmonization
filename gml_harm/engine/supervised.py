import torch
from typing import Any, Mapping, Dict

from .base_runner import BaseRunner


class SupervisedTrainer(BaseRunner):
    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        images = batch['images']
        masks = batch['masks']
        targets = batch['targets']

        model = self.model['model']
        output_dict: Dict[str, torch.Tensor] = model(images, masks)

        predicted_image = output_dict['predicted_image']
        harm_image = output_dict['harm_image']
        outputs = harm_image

        self.batch = {
            'predicted_images': predicted_image,
            'outputs': outputs,
            'targets': targets,
            'masks': masks,
            'targets_and_masks': (targets, masks)
        }

        if self.to_original_scale is not None:
            to_original_scale = self.to_original_scale
            outputs_255 = to_original_scale(outputs.detach())
            outputs_255 = torch.clip(outputs_255, 0, 255.)
            targets_255 = to_original_scale(targets)
            self.batch.update({
                'outputs_255': outputs_255,
                'targets_255': targets_255,
                'targets_1': targets_255 / 255.,
                'outputs_1': outputs_255 / 255.,
                'targets_and_masks_255': (targets_255, masks)
            })