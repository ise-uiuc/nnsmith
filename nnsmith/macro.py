import os

ONNX_EXTERNAL_DATA_DIR_SUFFIX = "-mlist"
NNSMITH_ORT_INTRA_OP_THREAD = int(os.getenv("NNSMITH_ORT_INTRA_OP_THREAD", 1))
NNSMITH_BUG_PATTERN_TOKEN = "${PATTERN}"


def onnx2external_data_dir(onnx_file):
    return onnx_file + ONNX_EXTERNAL_DATA_DIR_SUFFIX
