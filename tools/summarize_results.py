from ast import parse
import glob
import os.path as osp
import pandas as pd
import pickle as pkl
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./work_dirs")
    return parser.parse_args()

def main(args):
    results_pkl_files = glob.glob(osp.join(args))
    results
    for pkl_file in results_pkl_files:
        print("Found %s" % osp.basename(pkl_file))

        with open(pkl_file, "rb") as fi:
            data = pkl.load(fi)


if __name__ == '__main__':
    args = parse_args()
    main(args)