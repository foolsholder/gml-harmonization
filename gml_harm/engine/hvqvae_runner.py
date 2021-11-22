import torch
from catalyst import dl
from typing import Any, Mapping, Callable, Sequence


class HVQVAERunner(dl.Runner):
    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        content_alpha = batch['content_alpha']
        content_beta = batch['content_beta']

        reference_alpha = batch['reference_alpha']
        reference_beta = batch['reference_beta']

        content_quant_alpha, latent_loss_ca, \
            content_feat_alpha = self.model.encode_content(content_alpha)
        content_quant_beta, latent_loss_cb, \
            content_feat_beta = self.model.encode_content(content_beta)

        reference_quant_alpha, latent_loss_fa, \
            reference_feat_alpha = self.model.encode_reference(reference_alpha)
        reference_quant_beta, latent_loss_fb, \
            reference_feat_beta = self.model.encode_reference(reference_beta)

        _, _, content_appearance_feat_alpha = self.model.encode_reference(content_alpha)

        reconstruct_content_alpha = self.model.decode(content_alpha, reference_alpha)
        harmonize_content_alpha = self.model.decode(content_alpha, reference_beta)

        latent_loss_content = latent_loss_ca + latent_loss_cb
        latent_loss_reference = latent_loss_fa + latent_loss_fb

        self.batch_metrics.update({
            'latent_loss_content': latent_loss_content,
            'latent_loss_reference': latent_loss_reference
        })

        self.batch = {
            'content_alpha': content_alpha,
            'content_beta': content_beta,

            'reconstruct_content_alpha': reconstruct_content_alpha,
            'harmonize_content_alpha': harmonize_content_alpha,

            'content_feat_alpha': content_feat_alpha,
            'content_feat_beta': content_feat_beta,

            'reference_feat_alpha': reference_feat_alpha,
            'reference_feat_beta': reference_feat_beta,

            'content_appearance_feat_alpha': content_appearance_feat_alpha
        }