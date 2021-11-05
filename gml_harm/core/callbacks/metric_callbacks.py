from catalyst import dl
from typing import Optional, Union, Iterable, Dict



from ..metrics import MSEMetric, PSNRMetric, FNMSEMetric


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