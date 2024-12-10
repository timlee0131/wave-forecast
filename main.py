import argparse
from experiments import trainer

def get_args():
    parser = argparse.ArgumentParser(description="Heterogeneous Graph JEPA")

    parser.add_argument('-m', '--mode', choices=['train', 'aux', 'test'], type=str, default='train', help='Mode: train, aux, test')
    
    parser.add_argument('-d', '--dataset', choices=['waves', 'waves-2024', 'waves-51002'], type=str, default='waves', help='dataset to use')

    return parser.parse_args()

def main():
    args = get_args()

    if args.mode == 'aux':
        trainer.driver(args.dataset, aux=True)
    else:   # train by default
        trainer.driver(args.dataset)

if __name__ == '__main__':
    main()