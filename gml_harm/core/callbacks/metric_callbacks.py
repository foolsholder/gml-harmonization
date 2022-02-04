from catalyst import dl
from typing import Optional, Union, Iterable, Dict, OrderedDict as ORDType, Any

from ..metrics import MSEMetric, PSNRMetric, fMSEMetric, FNMSEMetric, IdentityMetric, LPIPSMetric
from ..losses.perceptual import ResNetPLMetric


class IdentityCallback(dl.FunctionalBatchMetricCallback):
    def __init__(self,
                 input_key: Union[str, Iterable[str], Dict[str, str]],
                 target_key: Union[str, Iterable[str], Dict[str, str]],
                 metric_key: str,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 compute_on_call: bool = True,
                 log_on_batch: bool = True):
        super(IdentityCallback, self).__init__(
            metric=IdentityMetric(prefix=prefix,
                                  suffix=suffix,
                                  compute_on_call=compute_on_call,
                                  metric_key=metric_key),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch)


class MSECallback(dl.FunctionalBatchMetricCallback):
    def __init__(self,
                 input_key: Union[str, Iterable[str], Dict[str, str]],
                 target_key: Union[str, Iterable[str], Dict[str, str]],
                 metric_key: str,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 compute_on_call: bool = True,
                 log_on_batch: bool = True):
        super(MSECallback, self).__init__(
            metric=MSEMetric(prefix=prefix,
                             suffix=suffix,
                             compute_on_call=compute_on_call,
                             metric_key=metric_key),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch)


class PSNRCallback(dl.FunctionalBatchMetricCallback):
    def __init__(self,
                 input_key: Union[str, Iterable[str], Dict[str, str]],
                 target_key: Union[str, Iterable[str], Dict[str, str]],
                 metric_key: str,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 compute_on_call: bool = True,
                 log_on_batch: bool = True, **kwargs):
        super(PSNRCallback, self).__init__(
            metric=PSNRMetric(prefix=prefix,
                              suffix=suffix,
                              compute_on_call=compute_on_call,
                              metric_key=metric_key,
                              **kwargs),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch)


class fMSECallback(dl.FunctionalBatchMetricCallback):
    def __init__(self,
                 input_key: Union[str, Iterable[str], Dict[str, str]],
                 target_key: Union[str, Iterable[str], Dict[str, str]],
                 metric_key: str,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 compute_on_call: bool = True,
                 log_on_batch: bool = True):
        super(fMSECallback, self).__init__(
            metric=fMSEMetric(prefix=prefix,
                               suffix=suffix,
                               compute_on_call=compute_on_call,
                               metric_key=metric_key),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch)


class FNMSECallback(dl.FunctionalBatchMetricCallback):
    def __init__(self,
                 input_key: Union[str, Iterable[str], Dict[str, str]],
                 target_key: Union[str, Iterable[str], Dict[str, str]],
                 metric_key: str,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 compute_on_call: bool = True,
                 log_on_batch: bool = True, **kwargs):
        super(FNMSECallback, self).__init__(
            metric=FNMSEMetric(prefix=prefix,
                               suffix=suffix,
                               compute_on_call=compute_on_call,
                               metric_key=metric_key,
                               **kwargs),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch)


class ResNetPLCallback(dl.FunctionalBatchMetricCallback):
    def __init__(self,
                 input_key: Union[str, Iterable[str], Dict[str, str]],
                 target_key: Union[str, Iterable[str], Dict[str, str]],
                 metric_key: str,
                 resnet_cfg: ORDType[str, Any],
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 compute_on_call: bool = True,
                 log_on_batch: bool = True, **kwargs):
        super(ResNetPLCallback, self).__init__(
            metric=ResNetPLMetric(prefix=prefix,
                                  suffix=suffix,
                                  compute_on_call=compute_on_call,
                                  metric_key=metric_key,
                                  resnet_cfg=resnet_cfg,
                                  **kwargs),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch)


class LPIPSCallback(dl.FunctionalBatchMetricCallback):
    def __init__(self,
                 input_key: Union[str, Iterable[str], Dict[str, str]],
                 target_key: Union[str, Iterable[str], Dict[str, str]],
                 metric_key: str,
                 lpips_model: str,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 compute_on_call: bool = True,
                 log_on_batch: bool = True, **kwargs):
        super(LPIPSCallback, self).__init__(
            metric=LPIPSMetric(prefix=prefix,
                               suffix=suffix,
                               compute_on_call=compute_on_call,
                               metric_key=metric_key,
                               lpips_model=lpips_model,
                               **kwargs),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch)