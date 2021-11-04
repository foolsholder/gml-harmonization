from catalyst import dl
from typing import Any, Mapping


class SupervisedTrainer(dl.Runner):
    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        images = batch['images']
        masks = batch['masks']
        targets = batch['targets']

        outputs = self.model(images, masks)
        self.batch = {
            'outputs': outputs,
            'targets': targets,
            'masks': masks,
            'targets_and_masks': (targets, masks)
        }