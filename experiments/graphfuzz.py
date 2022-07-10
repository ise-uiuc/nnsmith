"""
This is basically a simplied implementation of "Graph-Based Fuzz Testing for Deep Learning Inference Engine".
Here's a few facts:
1. They are not open-source.
2. We only implement how they generate a valid graph (basically ensure its capability).
3. (point 2 cont.) We don't care their complicated heuristics and corpus.
4. The key explanantion is in Section III.G "Shapes & parameters Calculator."
    4.1: In additoin to LEMON, they support non-unary operators, ensuring the shape match by:
        a. slicing;
        b. padding (keeping layers element-wise);
        c. unsqueezing; (we added this to make GraphFuzz even better)
    4.2: Operators with padding to try to ensure everything is element-wise.
        NOTE: Through the examples, they basically assume input rank is 4 (NCHW).

Thus we now conclude the search space:
- operators:
    - input/output rank allows: 4 or -1;
    - element-wise opeartors;
    - exceptions:
        pooling use padding to make it elementwise;
        convolution use padding to make it elementwise;
    - data type casting.
    - parameters shall be randomly genenrated;
- input:
    - single input; (they said MNN cannot support multi-input)
    - input rank: 4;
"""

import random
from typing import List
import warnings
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from nnsmith.util import mkdir
from nnsmith.abstract.op import (
    ALL_OP_TYPES,
    AbsOpBase,
    Softmax,
    BatchNorm2d,
    Concat,
    Input,
    Constant,
    NCHWConv2d,
    Pool2d,
    DType,
    Div,
    ElementWiseUnaryOp,
    BcastBinaryOp,
)
from nnsmith.dtype_test import rewrite_op_dtype


class GraphFuzzNet(torch.nn.Module):
    def __init__(self, op_list: List[AbsOpBase]) -> None:
        super().__init__()
        self.op_list = op_list
        self.torch_inst = []
        self.mlist = torch.nn.ModuleList()
        for op in op_list:
            inst = op.torch()
            self.torch_inst.append(inst)
            if isinstance(inst, torch.nn.Module):
                self.mlist.append(inst)

    def forward(self, x):
        available_tensors = [x]
        shape_checker = x.shape
        for idx, op in enumerate(self.op_list):
            n_inp = len(op.inp_ranks)
            n_out = len(op.out_ranks)
            selected = [random.choice(available_tensors) for _ in range(n_inp)]
            dtype_match = False
            for its in op.in_dtypes:
                n_match = 0
                for expected, have in zip(its, selected):
                    if str(expected) != str(have.dtype):
                        break
                    n_match += 1
                if n_match == n_inp:
                    dtype_match = True
                    break

            # dtype match
            if not dtype_match:
                type_req = random.choice(op.in_dtypes)
                selected = [
                    tensor.type(DType.torch(t)) for tensor, t in zip(selected, type_req)
                ]

            # rank match
            for i_st, st in enumerate(selected):
                while len(selected[i_st].shape) > len(shape_checker):
                    selected[i_st] = torch.select(selected[i_st], dim=0, index=0)

                while len(selected[i_st].shape) < len(shape_checker):
                    selected[i_st] = selected[i_st].unsqueeze(0)

            # shape match
            for i_st, st in enumerate(selected):
                assert len(st.shape) == len(shape_checker)
                if not st.shape == shape_checker:
                    pads = []
                    need2pad = False
                    need2slice = False
                    for i in range(len(st.shape)):
                        if st.shape[i] < shape_checker[i]:
                            pads.append(shape_checker[i] - st.shape[i])
                            need2pad = True
                        elif st.shape[i] > shape_checker[i]:
                            pads.append(0)
                            need2slice = True
                        else:
                            pads.append(0)

                    # padding if too small
                    if need2pad:
                        padding = []
                        for p in pads:
                            padding.append(0)
                            padding.append(p)
                        selected[i_st] = torch.nn.functional.pad(
                            selected[i_st], tuple(padding), mode="constant", value=1
                        )

                    # slicing if too large
                    if need2slice:
                        # must be rank of 4 now.
                        selected[i_st] = selected[i_st][
                            : shape_checker[0],
                            : shape_checker[1],
                            : shape_checker[2],
                            : shape_checker[3],
                        ]

            if isinstance(op, Div):
                # avoid float-point errors.
                selected[1] = selected[1].abs() + 0.1

            outs = self.torch_inst[idx](*selected)

            if n_out == 1:
                available_tensors.append(outs)
            else:
                available_tensors.extend(outs)

        return available_tensors[-1]


class GraphFuzz:
    @staticmethod
    def get_available_op_ts(try_all=False):
        # Using try all, we allow GraphFuzz to generate all possible operators.
        # even though most of it gonna fail.
        available_op_ts = []
        for op_t in ALL_OP_TYPES:
            if op_t is Input or op_t is Constant:
                continue
            if not try_all:
                if issubclass(op_t, ElementWiseUnaryOp):
                    available_op_ts.append(op_t)
                elif op_t is NCHWConv2d:
                    available_op_ts.append(op_t)
                elif issubclass(op_t, Concat):
                    available_op_ts.append(op_t)
                elif issubclass(op_t, Pool2d):
                    available_op_ts.append(op_t)
                elif issubclass(op_t, BcastBinaryOp):
                    available_op_ts.append(op_t)
                continue
            available_op_ts.append(op_t)
        return available_op_ts

    def get_pool2d_params(self, input_shape):
        """To make input output element-wise.
        Pool   (kw, kh, s, pad)
        Input  (iN, iC, iH, iW)
        Output (oN, oC, oH, oW)
        -------------------------------------
        oN = iN
        oC = iC
        oH = iH = (iH - kh + 2 * pad) / s + 1
        oW = iW = (iW - kw + 2 * pad) / s + 1
        0 <= pad <= (kh - 1) / 2 # The condition in the paper for Conv2d is not applicable for pooling.
        0 <= pad <= (kw - 1) / 2
        1 <= s <= Max_sH (let's say it's 4)
        1 <= s <= Max_sW (let's say it's 4)
        """
        assert len(input_shape) == 4
        iN, iC, iH, iW = input_shape
        __MAX_TRY__ = 256
        kh = None
        kw = None
        s = None
        pad = None
        for _ in range(__MAX_TRY__):
            kw = random.randint(1, iW)
            kh = random.randint(1, iH)
            s = random.randint(1, 4)
            pad = random.randint(0, (kh - 1) // 2)
            if (iH - kh + 2 * pad) / s + 1 == iH and (iW - kw + 2 * pad) / s + 1 == iW:
                return kh, kw, s, pad
        return [1, 1, 1, 0]  # simple fallback

    def get_conv2d_params(self, input_shape):
        assert len(input_shape) == 4
        iN, iC, iH, iW = input_shape
        __MAX_TRY__ = 256
        kh = None
        kw = None
        s = None
        pad = None
        for _ in range(__MAX_TRY__):
            kw = random.randint(1, iW)
            kh = random.randint(1, iH)
            s = random.randint(1, 4)
            pad = random.randint(0, kh - 1)
            if (iH - kh + 2 * pad) / s + 1 == iH and (iW - kw + 2 * pad) / s + 1 == iW:
                return iC, iC, kh, kw, s, pad
        return [iC, iC, 1, 1, 1, 0]  # simple fallback

    def __init__(
        self, approx_nop=10, dim_limit=[5, 5, 224, 224], try_all=False
    ) -> None:
        self.available_op_ts = self.get_available_op_ts(try_all=try_all)
        self.base_op_n = approx_nop
        self.dim_limit = dim_limit

    def create_random_schedule(self):
        """
        basically return a list of ops.
        """
        input_shape = []
        for ls in self.dim_limit:
            input_shape.append(random.randint(1, ls))

        ops = []
        op_ts = [random.choice(self.available_op_ts) for _ in range(self.base_op_n)]
        for op_t in op_ts:
            if op_t is NCHWConv2d:
                ic, oc, kw, kh, s, pad = self.get_conv2d_params(input_shape)
                ops.append(op_t(ic, oc, kh, kw, s, pad))
            elif issubclass(op_t, Pool2d):
                kw, kh, s, pad = self.get_pool2d_params(input_shape)
                ops.append(op_t(kh, kw, s, pad))
            elif op_t is Softmax:
                ops.append(op_t(random.randint(1, len(input_shape) - 1)))
            elif op_t is BatchNorm2d:
                ops.append(op_t(input_shape[1]))
            elif issubclass(op_t, Concat):
                op = op_t()
                op.extra_attrs["axis"] = random.randint(0, 3)
                ops.append(op)
            else:
                ops.append(
                    op_t(
                        *[
                            random.randint(0, 10)
                            for _ in range(op_t.get_num_var_param())
                        ]
                    )
                )

        return ops, input_shape

    def run_once(self, save_path=None):
        ops, ishape = self.create_random_schedule()
        model = GraphFuzzNet(ops)
        with torch.no_grad():
            torch.onnx.export(model, torch.randn(ishape), save_path, opset_version=14)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--approx_nop", type=int, default=10)
    parser.add_argument("--dim_limit", type=int, nargs="+", default=[5, 5, 224, 224])
    parser.add_argument("--time_budget", type=int, default=60 * 60 * 4)
    parser.add_argument("--onnx_dir", type=str, default=None)
    parser.add_argument("--ort_cache", type=str, default=None)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--try_all", action="store_true")
    args = parser.parse_args()

    print(f"Using seed {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mkdir(args.onnx_dir)

    if args.ort_cache:
        if not os.path.exists(args.ort_cache):
            print(f"Please first generate cache! (mkdir config first)")
            print(f"python nnsmith/dtype_test.py --cache {args.ort_cache}")
            exit(1)
        # must pre run this. otherwise using ort will slow down generation.
        rewrite_op_dtype(ALL_OP_TYPES, factory=None, cache=args.ort_cache)

    gf = GraphFuzz(
        approx_nop=args.approx_nop, dim_limit=args.dim_limit, try_all=args.try_all
    )

    # FORMAT: {generation time cost in seconds}, {model relative path}
    # MUST RANK by GENERATION ORDER.
    config_file = open(os.path.join(args.onnx_dir, "gentime.csv"), "w")

    start_time = time.time()
    gen_cnt = 0
    valid_cnt = 0

    with tqdm(total=args.time_budget) as pbar:
        while time.time() - start_time < args.time_budget:
            seed = random.getrandbits(32)
            to_name = f"{valid_cnt}.onnx"

            tstart = time.time()
            try:
                with warnings.catch_warnings():  # just shutup.
                    warnings.simplefilter("ignore")
                    gf.run_once(save_path=os.path.join(args.onnx_dir, to_name))
                label = to_name
                valid_cnt += 1
            except Exception as e:
                print(f"Fail when seed={seed}")
                print(e)
                label = "FAILURE"

            time_diff = time.time() - tstart
            config_file.write(f"{time_diff:.5f},{label}\n")

            gen_cnt += 1
            config_file.flush()

            pbar.update(int(time.time() - start_time) - pbar.n)
            pbar.set_description(f"valid={valid_cnt},fail={gen_cnt-valid_cnt}")
            pbar.refresh()
        config_file.close()
