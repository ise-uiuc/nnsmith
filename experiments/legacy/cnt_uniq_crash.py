import os

from nnsmith.backends import mk_factory, BackendFactory

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", type=str, help="One of ort, trt, tvm, and xla", required=True
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--root", type=str, help="Path to the bug folder")
    args = parser.parse_args()

    fac = mk_factory(args.backend, args.device, optmax=True)

    crash_msg = set()

    for dir in os.listdir(args.root):
        path = os.path.join(args.root, dir)
        if not os.path.isdir(path):
            continue
        if not dir.startswith("bug-"):
            continue
        if dir.startswith("bug-torch") or dir.startswith("bug-omin"):
            continue

        try:
            onnx_path = os.path.join(path, "model.onnx")
            onnx_model = BackendFactory.get_onnx_proto(onnx_path)
            try:
                fac.make_backend(onnx_model)
            except Exception as e:
                crash_msg.add(str(e))
        except Exception as e:
            print(e)  # filter model-too-large kind of things.
            continue

    print(f"{args.root} got {len(crash_msg)} different crash messages:")
    err_msg_path = os.path.join(args.root, "crash_msg.txt")
    print(f"Writing unique crash messages to {err_msg_path}")
    if crash_msg:
        with open(err_msg_path, "w") as f:
            for msg in crash_msg:
                print(msg, file=f)
                print("$\n", file=f)  # splitter
