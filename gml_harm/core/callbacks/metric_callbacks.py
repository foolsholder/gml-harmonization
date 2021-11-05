from catalyst import dl
from typing import Optional



from ..metrics import MSEMetric, PSNRMetric, FNMSEMetric


class MSECallback(dl.BatchMetricCallback):
    def __init__(self,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 **kwargs):
        super(MSECallback, self).__init__(
            metric=MSEMetric(prefix=prefix, suffix=suffix),
            **kwargs
        )


class PSNRCallback(dl.BatchMetricCallback):
    def __init__(self,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 **kwargs):
        super(PSNRCallback, self).__init__(
            metric=PSNRMetric(prefix=prefix, suffix=suffix),
            **kwargs
        )


class FNMSECallback(dl.BatchMetricCallback):
    def __init__(self,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 **kwargs):
        super(FNMSECallback, self).__init__(
            metric=FNMSEMetric(prefix=prefix, suffix=suffix),
            **kwargs
        )