from .kernel import ConvKernel, EMAKernel
from .ssm import SSMKernelDense, SSMKernelReal, SSMKernelDiag, SSMKernelDPLR

# Import FractionalS4DKernel from standalone s4d module
try:
    from models.s4.s4d import FractionalS4DKernel
    _has_fractional = True
except ImportError:
    _has_fractional = False

registry = {
    'conv': ConvKernel,
    'ema': EMAKernel,
    'dense': SSMKernelDense,
    'slow': SSMKernelDense,
    'real': SSMKernelReal,
    's4d': SSMKernelDiag,
    'diag': SSMKernelDiag,
    's4': SSMKernelDPLR,
    'nplr': SSMKernelDPLR,
    'dplr': SSMKernelDPLR,
}

# Add fractional kernel if available
if _has_fractional:
    registry['fractional'] = FractionalS4DKernel
    registry['fractional_s4d'] = FractionalS4DKernel
