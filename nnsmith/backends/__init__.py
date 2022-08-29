from nnsmith.backends.factory import BackendFactory


def mk_factory(name, device="cpu", optmax=True, **kwargs):
    if name == "ort" or name == "onnxruntime":
        from nnsmith.backends.onnxruntime import ORTFactory

        return ORTFactory(device=device, optmax=optmax, **kwargs)
    elif name == "tvm":
        from nnsmith.backends.tvm import TVMFactory

        # default executor is graph
        kwargs["executor"] = kwargs.get("executor", "graph")
        return TVMFactory(device=device, optmax=optmax, **kwargs)
    elif name == "trt":
        from nnsmith.backends.tensorrt import TRTFactory

        return TRTFactory()
    else:
        raise ValueError(f"unknown backend: {name}")
