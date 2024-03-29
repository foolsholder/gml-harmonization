import torch
import numpy as np

from typing import Optional
from catalyst import dl


def _get_grad_norm(model: torch.nn.Module):
    wn = 0.
    gn = 0.
    for par in model.parameters():
        wn += np.sum(par.detach().cpu().data.numpy().ravel() ** 2)
        if par.grad is not None:
            gn += np.sum(par.grad.detach().cpu().data.numpy().ravel() ** 2)
    return np.sqrt(wn), np.sqrt(gn)


class OptimizerCallback(dl.OptimizerCallback):
    def __init__(self, grad_clip_norm: Optional[float] = None, *args, **kwargs):
        super(OptimizerCallback, self).__init__(*args, **kwargs)
        self._grad_clip_norm = grad_clip_norm

    def on_batch_end(self, runner: "IRunner"):
        if runner.is_train_loader:
            self._accumulation_counter += 1
            need_gradient_step = self._accumulation_counter % self.accumulation_steps == 0

            loss = runner.batch_metrics[self.metric_key]
            runner.engine.backward_loss(loss, self.model, self.optimizer)

            wn, gn = _get_grad_norm(self.model)
            runner.batch_metrics['weight_norm_' + self.model_key] = wn
            runner.batch_metrics['grad_norm_' + self.model_key] = gn

            if self.grad_clip_fn is not None:
                self.grad_clip_fn(self.model.parameters())
            if self._grad_clip_norm is not None and gn > self._grad_clip_norm:
                torch.nn.utils.clip_grad_norm(self.model.parameters(),
                                              max_norm=self._grad_clip_norm,
                                              norm_type=2.)

            if need_gradient_step:
                runner.engine.optimizer_step(loss, self.model, self.optimizer)
                runner.engine.zero_grad(loss, self.model, self.optimizer)

        runner.batch_metrics.update(self._get_lr_momentum_stats())