import argparse
import random

from tqdm import tqdm

from nnsmith.graph_gen import random_model_gen, SymbolNet


def mknet(args):
    model_seed = random.getrandbits(32)
    gen, solution = random_model_gen(
        mode=args.mode, seed=model_seed, max_nodes=args.max_nodes, init_fp=True
    )
    net = SymbolNet(gen.abstract_graph, solution, alive_shapes=gen.alive_shapes)
    net.eval()
    return net, gen.num_op(), model_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_nodes", type=int, default=20)
    parser.add_argument("--n_model", type=int, default=1000)
    parser.add_argument("--mode", type=str, default="random")
    args = parser.parse_args()

    n_invalid = 0
    with tqdm(range(args.n_model)) as pbar:
        for model_id in pbar:
            net, num_op, model_seed = mknet(args)
            net.check_intermediate_numeric = True
            _ = net(*net.get_random_inps(base="center", margin=1))
            n_invalid += net.invalid_found_last
            cur_model_size = model_id + 1
            pbar.set_description(
                f"invalid:{n_invalid}/{cur_model_size} = {100 * n_invalid/cur_model_size:.2f}%"
            )
