from catalyst import metrics

from .functional import mse, psnr, fn_mse


class MSEMetric(metrics.FunctionalBatchMetric):
    def __init__(self, **kwargs):
        super(MSEMetric, self).__init__(
            metric_fn=mse,
            metric_key='mse',
            **kwargs
        )


class PSNRMetric(metrics.FunctionalBatchMetric):
    def __init__(self, **kwargs):
        super(PSNRMetric, self).__init__(
            metric_fn=psnr,
            metric_key='psnr',
            **kwargs
        )


class FNMSEMetric(metrics.FunctionalBatchMetric):
    def __init__(self, **kwargs):
        """
        kwargs['target_key'] == 'targets_and_masks'
        """
        super(FNMSEMetric, self).__init__(
            metric_fn=fn_mse,
            metric_key='fn_mse',
            **kwargs
        )