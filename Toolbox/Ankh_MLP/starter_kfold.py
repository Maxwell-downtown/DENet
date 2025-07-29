import argparse
import pandas as pd
import numpy as np
from model_kfold import V2

def main():
    
    v2 = V2(output_dir=args.output_dir,
                train_tsv=args.train,
                test_tsv=args.test,
                fasta=args.fasta,
                d_model=1536,
                save_log=args.save_log)
    
    if args.saved_model_dir:
        v2.ensemble_predict(args.saved_model_dir)
    elif args.resume_dir:
        v2.train_kfold(
            epochs=args.epochs, random_state=args.random,
            save_checkpoint=args.save_checkpoint, resume_path=args.resume_dir
        )
    else:
        v2.train_kfold(
            epochs=args.epochs, random_state=args.random,
            save_checkpoint=args.save_checkpoint
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', action='store', help='training data (TSV format)')
    parser.add_argument('--test', action='store', help='test data (TSV format)')
    parser.add_argument('--fasta', action='store', required=True, help='native sequence (FASTA format)')
    parser.add_argument('--epochs', action='store', type=int, default=200, help='total epochs')
    parser.add_argument('--random', action='store', type=int, default=42, help='random seed')
    parser.add_argument('--output_dir', action='store', help='directory to save model, prediction, etc.')
    parser.add_argument('--saved_model_dir', action='store', help='directory of trained models')
    parser.add_argument('--save_checkpoint', action='store_true', default=False, help='save pytorch model checkpoint')
    parser.add_argument('--save_prediction', action='store_true', default=False, help='save prediction')
    parser.add_argument('--resume_dir', action='store', help='directory of trained models to resume training')
    parser.add_argument('--save_log', action='store_true', default=False, help='save log file')
    args = parser.parse_args()
    main()


