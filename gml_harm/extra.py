import torch


def bn_update(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be trasferred to
            :attr:`device` before being passed into :attr:`model`.
    """
    if not _check_bn(model):
        return
    was_training = model.training
    model.train()
    momenta = {}
    model.to(device)
    model.apply(_reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    base = getattr(model, 'base')
    mask_fusion_bb = getattr(model, 'mask_fusion_bb')
    n = 0
    from tqdm.auto import tqdm
    for _ in range(1):
        for input in tqdm(loader):
            images = input['images']
            masks = input['masks']
            b = images.size(0)

            momentum = b / float(n + b)
            for module in momenta.keys():
                module.momentum = momentum

            if device is not None:
                images = images.to(device)
                masks = masks.to(device)

            with torch.no_grad():
                model(images, masks)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    model.train(was_training)


# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]
