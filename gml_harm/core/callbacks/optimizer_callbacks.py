import torch

from catalyst import dl


class OptimizerCallback(dl.OptimizerCallback):
    def on_batch_end(self, runner: "IRunner"):
        """Event handler."""
        if runner.is_train_loader:
            self._accumulation_counter += 1
            need_gradient_step = self._accumulation_counter % self.accumulation_steps == 0

            loss: torch.Tensor = runner.batch_metrics[self.metric_key]
            loss.backward()

            if self.grad_clip_fn is not None:
                self.grad_clip_fn(self.model.parameters())

            if need_gradient_step:
                self.optimizer.step()
                self.optimizer.zero_grad()

        runner.batch_metrics.update(self._get_lr_momentum_stats())