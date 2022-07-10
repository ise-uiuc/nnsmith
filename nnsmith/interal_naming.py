ONNX_EXTERNAL_DATA_DIR_SUFFIX = "-mlist"


def onnx2external_data_dir(onnx_file):
    return onnx_file + ONNX_EXTERNAL_DATA_DIR_SUFFIX
