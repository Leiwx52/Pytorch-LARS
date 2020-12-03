
import math
import re
import torch
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from collections import defaultdict

class Lars(Optimizer):
    r"""Implements LARS (Layer-wise Adaptive Rate Scaling).
    Layer-wise Adaptive Rate Scaling for large batch training.

    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)

    Implements the LARS learning rate scheme presented in the paper above. This
    optimizer is useful when scaling the batch size to up to 32K without
    significant performance degradation. It is recommended to use the optimizer
    in conjunction with:
        - Gradual learning rate warm-up
        - Linear learning rate scaling
        - Poly rule learning rate decay


    Args:
        net: Network that includes all params to be optimized. Note that
            the second args, `params` should be in `net`.
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        exclude_from_weight_decay: List of regex patterns of
              variables excluded from weight decay. Variables whose name
              contain a substring matching the pattern will be excluded.(default:['batchnorm', 'bias'])
        exclude_from_layer_adaptation: List of regex patterns of
              variables excluded from layer adaptation. Variables whose name
              contain a substring matching the pattern will be excluded. (default: None)
        ratio_clip_value: clamp trust_ratio in (0, ratio_clip_value) (default: 50)
              set to a high value to avoid it (e.g 10e3)
        decay_grad_clip_value: clamp grad(after weight_decay) in (-decay_grad_clip_value, decay_grad_clip_value) 
            (default: 50)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        epsilon (float, optional): epsilon to prevent zero division (default: 0)
    Example:
        # >>> optimizer = torch.optim.Lars(model, model.parameters(), lr=0.1, momentum=0.9)
        # >>> optimizer.zero_grad()
        # >>> loss_fn(model(input), target).backward()
        # >>> optimizer.step()
    
    Note:
        + Reference code: 
            #1 https://people.eecs.berkeley.edu/~youyang/lars_optimizer.py
            #2 https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py
            #3 https://github.com/fastalgo/imagenet_resnet50_lamb/blob/master/optimization.py
        
        + This implementation is different from some Pytorch optimizers, which does not need to pass a `net` argument. 
        This one adapt `exculde_from_weight_decay` and `exclude_from_layer_adaptation` by including this args.
        See Reference codes #1 #2 #3

        + The default values are set according to 

        + You can also implement a LARS optimizer without `net`-arg. Refer to this manner:
        https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/8f27c9b97d2ca7c6e05333d5766d144bf7d8c31b/train.py#L92
        You are welcome to open an issue or leave comments if you have better solutions.

    """

    def __init__(
        self, 
        net,
        params, 
        lr=required, 
        momentum=0, 
        eta=1e-3, 
        dampening=0,
        weight_decay=0, 
        exclude_from_weight_decay = ['batchnorm', 'bias'],
        exclude_from_layer_adaptation = None,
        ratio_clip_value = 50.,
        decay_grad_clip_value = 10.,
        nesterov=False, 
        epsilon=1e-5,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if eta < 0.0:
            raise ValueError("Invalid eta value: {}".format(eta))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if ratio_clip_value is not None and ratio_clip_value < 0.0:
            raise ValueError('Invalid ratio_clip_value: {}'.format(ratio_clip_value))
        if decay_grad_clip_value is not None and decay_grad_clip_value < 0.0:
            raise ValueError(
                'Invalid decay_grad_clip_value: {}'.format(decay_grad_clip_value)
            )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(lr=lr, momentum=momentum, eta=eta, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, epsilon=epsilon)

        self.net = net
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # `exclude_from_layer_adaptation` is set to be the same as
        # `exclude_from_weight_decay` if it is None.
        # Borrow from official tensorflow LAMB implementation
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay
        self.ratio_clip_value = ratio_clip_value
        self.decay_grad_clip_value = decay_grad_clip_value
        super(Lars, self).__init__(params, defaults)
        self._check()
        self._init_paraName()

    def __setstate__(self, state):
        super(Lars, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
    
    def _check(self):
        r'''
        Check if all `params` are in `net`
        '''
        netDict = defaultdict(dict)
        for p in self.net.parameters():
            netDict[p]=True
        for group in self.param_groups:
            for p in group['params']:
                if netDict.get(p) is None:
                    msg = (
                        'All `params` must be in `net` '
                        'but got unexpected parameter(s). '
                        'Please check.' 
                    )
                    raise RuntimeError(msg)
        del netDict

    def _init_paraName(self):
        r'''
        Get all parameters' name in `self.net` and then store it in `self.state`
        Do it in initialzation.
        '''
        for name,para in self.net.named_parameters():
            module_top2bottom = name.split('.')
            cursor = self.net
            for i in range(len(module_top2bottom)-1):
                cursor = getattr(cursor, module_top2bottom[i])
            bottom_m_name = repr(cursor).split('(')[0]
            this_para_name = '.'.join([bottom_m_name, module_top2bottom[-1]])
            
            # Adding name for each parameter
            # e.g. Conv2d.weight, BatchNorm2d.bias, etc.
            # This is for `exclude_weight_dacay` and `exculde_layer_adaptation`
            self.state[para]['para_name'] = this_para_name

    def _do_layer_adaptation(self, para):
        r"""
        Whether to do layer-wise learning rate adaptation for `para`.
        """
        para_name = self.state[para]['para_name']
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, para_name, re.I) is not None:
                    return False
        return True
    
    def _do_use_weight_decay(self, para):
        r"""Whether to use L2 weight decay for `param`."""
        para_name = self.state[para]['para_name']
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, para_name, re.I) is not None:
                    return False
        return True

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            dampening = group['dampening']
            nesterov = group['nesterov']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                if self._do_layer_adaptation(p):
                    wd_coff = weight_decay if self._do_use_weight_decay(p) else 1.0
                    w_norm = torch.norm(p.data)
                    g_norm = torch.norm(p.grad.data)
                    if w_norm * g_norm > 0:
                        trust_ratio = eta * w_norm / (g_norm +
                            wd_coff * w_norm + epsilon)
                    else:
                        trust_ratio = 1
                else:
                    trust_ratio = 1
                if self.ratio_clip_value is not None:
                    trust_ratio = min(trust_ratio, self.ratio_clip_value)
                
                d_p = p.grad.data
                if weight_decay != 0 and self._do_use_weight_decay(p):
                    d_p.add_(weight_decay, p.data)
                if self.decay_grad_clip_value is not None:
                    d_p.clamp_(-self.decay_grad_clip_value, self.decay_grad_clip_value)
                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-trust_ratio * group['lr'], d_p)

        return loss

if __name__ == "__main__":
    '''
    A toy model.
    '''
    import torchvision
    resnet = torchvision.models.resnet18(pretrained=False)
    optim = Lars(
            resnet, 
            resnet.parameters(), 
            lr=0.01, 
            momentum=0.99, 
            eta=1e-3, 
            dampening=0,
            weight_decay=0.001, 
            exclude_from_weight_decay = ['bias','batchnorm'],
            exclude_from_layer_adaptation = None,
            ratio_clip_value = 50.,
            decay_grad_clip_value = 10.,
        )
    criterion = torch.nn.CrossEntropyLoss()

    resnet.zero_grad()
    inp = torch.randn(1,3,224,224)
    outp = resnet(inp)
    target = torch.ones(1,).long()
    loss = criterion(outp, target)
    loss.backward()
    optim.step()

    # check parameters if they use `weight_decay` or `layerwise adaptation`
    state = optim.state
    for group in optim.param_groups:
        for p in group['params']:
            print('[{}] : weight_decay ({}) | layerwise adaptation ({}).'.format(
                state[p]['para_name'], optim._do_use_weight_decay(p), optim._do_use_weight_decay(p)
            ))