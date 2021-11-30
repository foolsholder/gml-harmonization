import torch
from catalyst import dl
from typing import Any, Mapping, Callable, Sequence

from ..data.transforms import ToOriginalScale


class HVQVAERunner(dl.Runner):
    def __init__(self, restore_scale=True):
        super(HVQVAERunner, self).__init__()
        self.to_original_scale = None
        if restore_scale is not None:
            self.to_original_scale = ToOriginalScale()

    def handle_batch_train(self, batch: Mapping[str, Any]) -> None:
        content_alpha = batch['content_alpha']
        content_beta = batch['content_beta']

        reference_alpha = batch['reference_alpha']
        reference_beta = batch['reference_beta']

        content_enc = self.model['content_encoder']
        reference_enc = self.model['reference_encoder']
        decoder = self.model['decoder']

        content_quant_alpha, latent_loss_ca, \
        content_feat_alpha = content_enc(content_alpha)
        content_quant_beta, latent_loss_cb, \
        content_feat_beta = content_enc(content_beta)

        reference_quant_alpha, latent_loss_fa, \
        reference_feat_alpha = reference_enc(reference_alpha)
        reference_quant_beta, latent_loss_fb, \
        reference_feat_beta = reference_enc(reference_beta)

        _, _, content_appearance_feat_alpha = reference_enc(content_alpha)

        reconstruct_content_alpha = decoder(content_quant_alpha, reference_quant_alpha)
        harmonize_content_alpha = decoder(content_quant_alpha, reference_quant_beta)

        latent_loss_content = latent_loss_ca + latent_loss_cb
        latent_loss_reference = latent_loss_fa + latent_loss_fb

        self.batch = {
            'content_alpha': content_alpha,
            'content_beta': content_beta,

            'reconstruct_content_alpha': reconstruct_content_alpha,
            'harmonize_content_alpha': harmonize_content_alpha,

            'content_feat_alpha': content_feat_alpha,
            'content_feat_beta': content_feat_beta,

            'reference_feat_alpha': reference_feat_alpha,
            'reference_feat_beta': reference_feat_beta,

            'content_appearance_feat_alpha': content_appearance_feat_alpha,

            'latent_loss_content': latent_loss_content.mean(),
            'latent_loss_reference': latent_loss_reference.mean()
        }

    def handle_batch_valid(self, batch: Mapping[str, Any]) -> None:
        content = batch['content_images']
        reference = batch['reference_images']
        masks = batch['masks']
        targets = batch['targets']

        model = self.model['model']

        harm_content, latent_loss = model(content, reference)

        outputs = content * masks + (1. - masks) * reference

        self.batch = {
            'outputs': outputs,
            'targets': targets,
            'masks': masks,
            'targets_and_masks': (targets, masks),
            'latent_loss': latent_loss.mean()
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

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        if self.is_train_loader:
            self.handle_batch_train(batch)
        elif self.is_valid_loader:
            self.handle_batch_valid(batch)