import argparse
from experiments import trainer

def get_args():
    parser = argparse.ArgumentParser(description="Heterogeneous Graph JEPA")

    """
    Arguments for program modes
    train: train the model
    aux: auxiliary functions (loss analysis, station wave information, etc.)
    test: test the model
    dat: data engineering (creating new pt files from raw data, etc.)
    """
    parser.add_argument('-m', '--mode', choices=['train', 'aux', 'test', 'dat'], type=str, default='train', help='Mode: train, aux, test, dat')
    
    parser.add_argument('-d', '--dataset', choices=['waves', 'waves-2024', 'waves-51002', 'waves-51002-2016', 'waves-51002-2017', 'waves-51002-2018'], type=str, default='waves-51002-2017', help='dataset to use')
    # parser.add_argument('-d', '--dataset', choices=['waves', 'gnn-1'], type=str, default='gnn-1', help='dataset to use')

    return parser.parse_args()

def main():
    args = get_args()

    if args.mode == 'aux':
        trainer.driver(args.dataset, aux=True)
    else:   # train by default
        trainer.driver(args.dataset)

if __name__ == '__main__':
    main()