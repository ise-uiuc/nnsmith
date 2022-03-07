import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tvm_home', type=str, required=True,
                        help='Path to the tvm home.')
    parser.add_argument('--src_cov_report', type=str, required=True,
                        help='The report file after fuzzing. '
                        'see https://clang.llvm.org/docs/SanitizerCoverage.html#disabling-instrumentation-without-source-modification')
    parser.add_argument('--output', type=str, default='tvm-blocklist.txt')
    args = parser.parse_args()

    tvm_home = os.path.realpath(args.tvm_home)
    
    block_list_content = \
R"""# Repeated std library is noisy.
src:/usr/lib*
src:/usr/include/*
# 3rdparty library is noisy.
""" + f'src:{tvm_home}/3rdparty/*\n'

    cov_info = open(args.src_cov_report, 'r').readlines()[2:]
    matches = ['.h', '.hpp', '.hxx', '.c', '.cc', '.cpp', '.cxx', '.cu', '.cuh']
    for line in cov_info:
        parts = line.split()
        if len(parts) == 0 or not any(x in parts[0] for x in matches):
            continue # invalid line.

        if '3rdparty' in parts[0]: # nvm. already filtered.
            continue
        
        if parts[3] == '0.00%': # not covered.
            block_list_content += f'src:{tvm_home}/{parts[0]}\n'

    with open(args.output, 'w') as f:
        f.write(block_list_content)
