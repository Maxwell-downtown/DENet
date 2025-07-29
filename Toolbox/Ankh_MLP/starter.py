import argparse
import pandas as pd
import numpy as np
from model import V1

def main():

    v1 = V1(output_dir=args.output_dir,
                train_tsv=args.train,
                test_tsv=args.test,
                fasta=args.fasta,
                split_ratio=args.split_ratio,
                n_ensembles=args.n_ensembles,
                batch_size=args.batch_size, save_log=args.save_log)
    
    if args.saved_model_dir:
        v1.load_checkpoint(args.saved_model_dir)
    else:
        v1.train(
            epochs=args.epochs, patience=args.patience,
            log_freq=args.log_freq, eval_freq=args.eval_freq,
            save_checkpoint=args.save_checkpoint
        )
    test_results = v1.test(
        model_label='Test', mode='ensemble',
        save_prediction=args.save_prediction,
    )
    test_res_msg = 'Testing Ensemble Model: Loss: {:.4f}\t'.format(test_results['loss'])
    test_res_msg += '\t'.join(['Test {}: {:.6f}'.format(k, v) for (k, v) in test_results['metric'].items()])
    v1.logger.info(test_res_msg + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', action='store', help='training data (TSV format)')
    parser.add_argument('--test', action='store', help='test data (TSV format)')
    parser.add_argument('--fasta', action='store', required=True, help='native sequence (FASTA format)')
    parser.add_argument('--n_ensembles', action='store', type=int, default=1, help='number of models in ensemble')
    parser.add_argument('--split_ratio', action='store', type=float, nargs='+', default=[0.7, 0.1, 0.2],
                        help='ratio to split training data. [train, valid] or [train, valid, test]')
    parser.add_argument('--epochs', action='store', type=int, default=500, help='total epochs')
    parser.add_argument('--patience', action='store', type=int, help='patience for early stopping')
    parser.add_argument('--batch_size', action='store', type=int, default=128, help='batch size')
    parser.add_argument('--log_freq', action='store', type=int, default=100,
                        help='logging for this many epochs')
    parser.add_argument('--eval_freq', action='store', type=int, default=50,
                        help='evaluate (on validation set) for this many epochs')
    parser.add_argument('--output_dir', action='store', help='directory to save model, prediction, etc.')
    parser.add_argument('--saved_model_dir', action='store', help='directory of trained models')
    parser.add_argument('--save_checkpoint', action='store_true', default=False, help='save pytorch model checkpoint')
    parser.add_argument('--save_prediction', action='store_true', default=False, help='save prediction')
    parser.add_argument('--save_log', action='store_true', default=False, help='save log file')
    args = parser.parse_args()
    main()


