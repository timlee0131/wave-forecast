import argparse
from experiments import trainer, stgnn_trainer

def get_args():
    parser = argparse.ArgumentParser(description="Nueral Forecasting of Ocean Waves")

    """
    Initial Arg
    --cat
        gnn: ST Graph Neural Network
        solo: Individual models (Simple Linear, MLP, CNN)
    --dataset
        GNN datasets
        icbn: Irish Coast Buoy Network
        labn: Los Angeles Buoy Network
        
        Individual datasets:
        waves: Waves dataset (icbn separate)
        waves-2024: Waves dataset with 2024 data
        waves-51002: Waves dataset with station 51002
        waves-51002-2016: Waves dataset with station 51002 from 2016
        waves-51002-2017: Waves dataset with station 51002 from 2017
        waves-51002-2018: Waves dataset with station 51002 from 2018
    --mode
        train: train the model
        aux: auxiliary functions (loss analysis, station wave information, etc.)
        test: test the model
        dat: data engineering (creating new pt files from raw data, etc.)
    """
    
    sub_parser = parser.add_subparsers(dest='cat')
    
    gnn_parser = sub_parser.add_parser('gnn')
    gnn_parser.add_argument('-d', '--dataset', choices=['icbn', 'labn'], type=str, default='icbn', help='dataset to use')
    gnn_parser.add_argument('-m', '--mode', choices=['train', 'aux', 'test', 'dat'], type=str, default='train', help='Mode: train, aux, test, dat')
    
    solo_parser = sub_parser.add_parser('solo')    
    # solo_parser.add_argument('-d', '--dataset', choices=['waves', 'waves-2024', 'waves-51002', 'waves-51002-2016', 'waves-51002-2017', 'waves-51002-2018'], type=str, default='waves-51002-2017', help='dataset to use')
    solo_parser.add_argument('-d', '--dataset', choices=['s46025', 's46069', 's46219', 's46221', 's46251', 'OneForAll'], type=str, default='s46221', help='dataset to use')
    solo_parser.add_argument('-m', '--mode', choices=['train', 'aux', 'test', 'dat'], type=str, default='train', help='Mode: train, aux, test, dat')

    return parser.parse_args()

def main():
    args = get_args()
    
    if args.cat == 'gnn':
        stgnn_trainer.driver(args.dataset)
    elif args.cat == 'solo':
        if args.mode == 'aux':
            trainer.driver(args.dataset, aux=True)
        else:   # train by default
            trainer.driver(args.dataset)
    else:
        raise ValueError('Invalid category')

if __name__ == '__main__':
    main()