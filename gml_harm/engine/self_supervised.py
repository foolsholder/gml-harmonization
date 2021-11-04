import torch
from catalyst import dl
from typing import Any, Mapping, Callable, Sequence


class SelfSupervisedTrainer(dl.Runner):
    def __init__(self, feat_aggregation: Callable[[Sequence[torch.Tensor],
                                                   Sequence[torch.Tensor]],
                                                  torch.Tensor]) -> None:
        super(SelfSupervisedTrainer, self).__init__()
        self._feat_aggr = feat_aggregation

    def _extract_enc_feats(self, batch: Mapping[str, Any]):
        content_alpha = batch['content_alpha']
        content_beta = batch['content_beta']

        reference_alpha = batch['reference_alpha']
        reference_beta = batch['reference_beta']

        feat_cont_alpha = self.model['cont_enc'](content_alpha)
        feat_cont_beta = self.model['cont_enc'](content_beta)

        feat_ref_alpha = self.model['ref_enc'](reference_alpha)
        feat_ref_beta = self.model['ref_enc'](reference_beta)

        return feat_cont_alpha, feat_cont_beta, feat_ref_alpha, feat_ref_beta

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        content_alpha = batch['content_alpha']
        content_beta = batch['content_beta']

        feat_cont_alpha, feat_cont_beta, feat_ref_alpha, feat_ref_beta = self._extract_enc_feats(batch)

        reconstruct_content_alpha = self._feat_aggr(feat_ref_alpha, feat_cont_alpha)
        reconstruct_content_alpha = self.model['decoder'](reconstruct_content_alpha)

        harmonize_content_alpha = self._feat_aggr(feat_ref_beta, feat_cont_alpha)
        harmonize_content_alpha = self.model['decoder'](harmonize_content_alpha)

        self.batch = {
            'content_alpha': content_alpha,
            'content_beta': content_beta,

            'reconstruct_content_alpha': reconstruct_content_alpha,
            'harmonize_content_alpha': harmonize_content_alpha,

            'feat_cont_alpha': feat_cont_alpha,
            'feat_cont_beta': feat_cont_beta,

            'feat_ref_alpha': feat_ref_alpha,
            'feat_ref_beta': feat_ref_beta,
        }