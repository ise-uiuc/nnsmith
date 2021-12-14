from nnsmith.graph_input_gen import gen_model_and_range_safe


if __name__ == '__main__':
    for i in range(5):
        print(gen_model_and_range_safe(
            './output.onnx', seed=i, max_node_size=20, max_gen_millisec=10)[1:])
    print('pass')
