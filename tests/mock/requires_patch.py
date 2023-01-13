from nnsmith.abstract.arith import nnsmith_lt
from nnsmith.abstract.extension import patch_requires


@patch_requires("global", "core.NCHWConv2d")
def limit_conv2d(self, _):
    # let the kernels to be > 3
    return [nnsmith_lt(3, self.kernel_h_size), nnsmith_lt(3, self.kernel_w_size)]
