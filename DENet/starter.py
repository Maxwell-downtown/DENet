import argparse
import pandas as pd
import numpy as np
from denet import DENet


# import tbe structure contact map for GNN edge:
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def main():
    c_maps = []
    for i in range(0, 1):
        structure_csv = args.structure + str(i) + '.csv'
        contact_map = pd.read_csv(structure_csv, header=None)
        contact_map = contact_map.to_numpy()
        contact_map = contact_map[:, :]
        contact_map = normalize(contact_map)
        c_map = contact_map + np.eye(contact_map.shape[0])
        c_map = normalize(c_map)
        #c_map = torch.from_numpy(c_map).float()
        c_maps.append(c_map)
    denet = DENet(output_dir=args.output_dir,
                train_tsv=args.train,
                test_tsv=args.test,
                fasta=args.fasta,
                comutation=args.comutation,
                d_model=args.d_local, d_h=args.d_h, d_out=args.d_out,
                use_loc_feat=(not args.no_comutation),
                use_glob_feat=(not args.no_PLM),
                use_gcn=args.use_gcn,
                c_map=c_maps, split_ratio=args.split_ratio,
                n_ensembles=args.n_ensembles,
                batch_size=args.batch_size, save_log=args.save_log)
    
    if args.saved_model_dir:
        denet.load_checkpoint(args.saved_model_dir)
    else:
        denet.train(
            epochs=args.epochs, patience=args.patience,
            log_freq=args.log_freq, eval_freq=args.eval_freq,
            save_checkpoint=args.save_checkpoint
        )
    test_results = denet.test(
        model_label='Test', mode='ensemble',
        save_prediction=args.save_prediction,
    )
    test_res_msg = 'Testing Ensemble Model: Loss: {:.4f}\t'.format(test_results['loss'])
    test_res_msg += '\t'.join(['Test {}: {:.6f}'.format(k, v) for (k, v) in test_results['metric'].items()])
    denet.logger.info(test_res_msg + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', action='store', help='training data (TSV format)')
    parser.add_argument('--test', action='store', help='test data (TSV format)')
    parser.add_argument('--fasta', action='store', required=True, help='native sequence (FASTA format)')
    parser.add_argument('--comutation', action='store', help='co-mutation information from DE or MSA(braw format)')
    parser.add_argument('--structure', action='store', help='contact map directory(csv format)')
    parser.add_argument('--no_comutation', action='store_true', help='do not use co-mutation information')
    parser.add_argument('--no_PLM', action='store_true', help='do not use PLM embedding')
    parser.add_argument('--split_ratio', action='store', type=float, nargs='+', default=[0.7, 0.1, 0.2],
                        help='ratio to split training data. [train, valid] or [train, valid, test]')
    parser.add_argument('--n_ensembles', action='store', type=int, default=3, help='number of models in ensemble')
    parser.add_argument('--epochs', action='store', type=int, default=1000, help='total epochs')
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
    parser.add_argument('--d_h', action='store', type=int, default=32, help='hidden dimension')
    parser.add_argument('--d_local', action='store', type=int, default=256, help='local dimension')
    parser.add_argument('--d_out', action='store', type=int, default=256, help='GCN out dimension')
    parser.add_argument('--use_gcn', action='store_true', default=False, help='use gcn or not')
    args = parser.parse_args()
    main()


