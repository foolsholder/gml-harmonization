import torch

from catalyst import metrics
from copy import copy

from .functional import mse, psnr, fn_mse


class MSEMetric(metrics.FunctionalBatchMetric):
    def __init__(self, metric_key, **kwargs):
        super(MSEMetric, self).__init__(
            metric_fn=mse,
            metric_key=metric_key,
            **kwargs
        )


class PSNRMetric(metrics.FunctionalBatchMetric):
    def __init__(self, metric_key, max_pixel_value=255.0, **kwargs):
        super(PSNRMetric, self).__init__(
            metric_fn=psnr,
            metric_key=metric_key,
            **kwargs
        )
        self.max_pixel_value = max_pixel_value

    def update(self, batch_size: int, *args, **kwargs) -> torch.Tensor:
        """
        Calculate metric and update average metric
        Args:
            batch_size: current batch size for metric statistics aggregation
            *args: args for metric_fn
            **kwargs: kwargs for metric_fn
        Returns:
            custom metric
        """
        kwargs.update({'max_pixel_value': self.max_pixel_value})
        value = self.metric_fn(*args, **kwargs)
        self.additive_metric.update(float(value), batch_size)
        return value


class FNMSEMetric(metrics.FunctionalBatchMetric):
    def __init__(self, metric_key, min_area = 100.0, **kwargs):
        """
        kwargs['target_key'] == 'targets_and_masks'
        """
        super(FNMSEMetric, self).__init__(
            metric_fn=fn_mse,
            metric_key=metric_key,
            **kwargs
        )
        self.min_area = min_area

    def update(self, batch_size: int, *args, **kwargs) -> torch.Tensor:
        """
        Calculate metric and update average metric
        Args:
            batch_size: current batch size for metric statistics aggregation
            *args: args for metric_fn
            **kwargs: kwargs for metric_fn
        Returns:
            custom metric
        """
        kwargs.update({'min_area': self.min_area})
        value = self.metric_fn(*args, **kwargs)
        self.additive_metric.update(float(value), batch_size)
        return value
