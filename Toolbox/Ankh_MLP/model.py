import torch
import torch.optim as optim
from tqdm import tqdm
import scipy.stats
import pathlib
import copy
import time

from data import Dataset
from A3_1 import Predictor
import torch.nn.functional as F
from utils import Saver, EarlyStopping, Logger


class V1(object):
    def __init__(self,
                 output_dir=None,
                 train_tsv=None,
                 test_tsv=None,
                 fasta=None,
                 split_ratio=None,
                 random_seed=42,
                 n_ensembles=1,
                 d_model=1536, d_h=128, d_attn=3,
                 batch_size=3, save_log=False):
        self.dataset = Dataset(
            train_tsv=train_tsv, test_tsv=test_tsv, fasta=fasta,
            split_ratio=split_ratio,
            random_seed=random_seed)

        self.seed = random_seed
        self.saver = Saver(output_dir=output_dir)
        self.logger = Logger(logfile=self.saver.save_dir / 'exp.log' if save_log else None)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [Predictor(
            out_dim=d_model, d_h=d_h, d_attn=d_attn
        ).to(self.device) for _ in range(n_ensembles)]

        self.criterion = F.mse_loss
        self.batch_size = batch_size
        self.optimizers = [optim.Adam(model.parameters()) for model in self.models]
        self._test_pack = None

    @property
    def test_pack(self):
        if self._test_pack is None:
            test_loader, test_df = self.dataset.get_dataloader(
                'test', batch_size=self.batch_size, return_df=True)
            self._test_pack = (test_loader, test_df)
        return self._test_pack

    @property
    def test_loader(self):
        return self.test_pack[0]

    @property
    def test_df(self):
        return self.test_pack[1]

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        if not checkpoint_dir.is_dir():
            raise ValueError(f'{checkpoint_dir} is not a directory')
        for i in range(len(self.models)):
            checkpoint_path = f'{checkpoint_dir}/model_{i + 1}.pt'
            self.logger.info('Load pretrained model from {}'.format(checkpoint_path))
            pt = torch.load(checkpoint_path, weights_only=False)
            model_dict = self.models[i].state_dict()
            model_pretrained_dict = {k: v for k, v in pt['model_state_dict'].items() if k in model_dict}
            model_dict.update(model_pretrained_dict)
            self.models[i].load_state_dict(model_dict)
            self.optimizers[i].load_state_dict(pt['optimizer_state_dict'])

    def load_single_pretrained_model(self, checkpoint_path, model=None, optimizer=None, is_resume=False):
        self.logger.info('Load pretrained model from {}'.format(checkpoint_path))
        pt = torch.load(checkpoint_path, weights_only=False)
        model_dict = model.state_dict()
        model_pretrained_dict = {k: v for k, v in pt['model_state_dict'].items() if k in model_dict}
        model_dict.update(model_pretrained_dict)
        model.load_state_dict(model_dict)
        optimizer.load_state_dict(pt['optimizer_state_dict'])
        return (model, optimizer, pt['log_info']) if is_resume else (model, optimizer)
    
    def save_checkpoint(self, ckp_name=None, model_dict=None, opt_dict=None, log_info=None):
        ckp = {'model_state_dict': model_dict, 'optimizer_state_dict': opt_dict}
        ckp['log_info'] = log_info
        self.saver.save_ckp(ckp, ckp_name)

    def train(self, epochs=1000, log_freq=100, eval_freq=50, patience=500, save_checkpoint=False, resume_path=None):
        assert eval_freq <= log_freq
        monitoring_score = 'corr'
        for midx, (model, optimizer) in enumerate(zip(self.models, self.optimizers), start=1):
            (train_loader, train_df), (valid_loader, valid_df) = \
                self.dataset.get_dataloader(
                    'train_valid', self.batch_size,
                    return_df=True, resample_train_valid=True)
            print(self.seed)
            
            if resume_path is not None:
                model, optimizer, log_info = self.load_single_pretrained_model(
                    '{}/model_{}.pt'.format(resume_path, midx),
                    model=model, optimizer=optimizer, is_resume=True)
                start_epoch = log_info['epoch'] + 1
                best_score = log_info['best_{}'.format(monitoring_score)]
            else:
                start_epoch = 1
                best_score = None

            best_model_state_dict = None
            stopper = EarlyStopping(patience=patience, eval_freq=eval_freq, best_score=best_score)
            model.train()
            try:
                for epoch in range(start_epoch, epochs + 1):
                    time_start = time.time()
                    tot_loss = 0
                    for step, batch in tqdm(enumerate(train_loader, 1),
                        leave=False, desc=f'M-{midx} E-{epoch}', total=len(train_loader)):
                        y = batch['label'].to(self.device)
                        X = batch['glob_feat'].to(self.device)
                        optimizer.zero_grad()
                        output = model(X)
                        output = output.view(-1)
                        loss = self.criterion(output, y)

                        loss.backward()
                        optimizer.step()
                        tot_loss += loss.item()

                    if epoch % eval_freq == 0:
                        val_results = self.test(test_model=model, test_loader=valid_loader,
                            test_df=valid_df, mode='val')
                        model.train()
                        is_best = stopper.update(val_results['metric'][monitoring_score])
                        if is_best:
                            best_model_state_dict = copy.deepcopy(model.state_dict())
                            if save_checkpoint:
                                self.save_checkpoint(ckp_name='model_{}.pt'.format(midx),
                                                     model_dict=model.state_dict(),
                                                     opt_dict=optimizer.state_dict(),
                                                     log_info={
                                                         'epoch': epoch,
                                                         'best_{}'.format(monitoring_score): stopper.best_score,
                                                         'val_loss': val_results['loss'],
                                                         'val_results': val_results['metric']
                                                     })


                    if epoch % log_freq == 0:
                        train_results = self.test(test_model=model, test_loader=train_loader,
                                test_df=train_df, mode='val')
                        if (log_freq <= eval_freq) or (log_freq % eval_freq != 0):
                            val_results = self.test(test_model=model, test_loader=valid_loader,
                                test_df=valid_df, mode='val')
                        model.train()
                        self.logger.info(
                            'Model: {}/{}'.format(midx, len(self.models))
                            + '\tEpoch: {}/{}'.format(epoch, epochs)
                            + '\tTrain loss: {:.4f}'.format(tot_loss / step)
                            + '\tVal loss: {:.4f}'.format(val_results['loss'])
                            + '\t' + '\t'.join(['Val {}: {:.4f}'.format(k, v) \
                                    for (k, v) in val_results['metric'].items()])
                            + '\tBest {n}: {b:.4f}\t'.format(n=monitoring_score, b=stopper.best_score)
                            + '\t{:.1f} s/epoch'.format(time.time() - time_start)
                            )
                        time_start = time.time()

                    if stopper.early_stop:
                        self.logger.info('Early stop at epoch {}'.format(epoch))
                        break
            except KeyboardInterrupt:
                self.logger.info('Exiting model training from keyboard interrupt')
            if best_model_state_dict is not None:
                model.load_state_dict(best_model_state_dict)

            test_results = self.test(test_model=model, model_label='model_{}'.format(midx))
            test_res_msg = 'Testing Model {}: Loss: {:.4f}\t'.format(midx, test_results['loss'])
            test_res_msg += '\t'.join(['Test {}: {:.6f}'.format(k, v) \
                                for (k, v) in test_results['metric'].items()])
            self.logger.info(test_res_msg + '\n')

    def test(self, test_model=None, test_loader=None, test_df=None,
                checkpoint_dir=None, save_prediction=False,
                calc_metric=True, calc_loss=True, model_label=None, mode='test'):
        if checkpoint_dir is not None:
            self.load_pretrained_model(checkpoint_dir)
        if test_loader is None and test_df is None:
            test_loader = self.test_loader
            test_df = self.test_df
        test_models = self.models if test_model is None else [test_model]
        esb_ypred, esb_yprob = None, None
        esb_loss = 0
        for model in test_models:
            model.eval()
            y_true, y_pred, y_prob = None, None, None
            tot_loss = 0
            with torch.no_grad():
                for step, batch in tqdm(enumerate(test_loader, 1),
                        desc=mode, leave=False, total=len(test_loader)):
                    
                    X = batch['glob_feat'].to(self.device)
                    output = model(X)
                    output = output.view(-1)
                    if calc_loss:
                        y = batch['label'].to(self.device)
                        loss = self.criterion(output, y)
                        tot_loss += loss.item()
                    y_pred = output if y_pred is None else torch.cat((y_pred, output), dim=0)
            y_pred = y_pred.detach().cpu() if self.device == torch.device('cuda') else y_pred.detach()
            esb_ypred = y_pred.view(-1, 1) if esb_ypred is None else torch.cat((esb_ypred, y_pred.view(-1, 1)), dim=1)
            esb_loss += tot_loss / step

        esb_ypred = esb_ypred.mean(axis=1).numpy()
        esb_loss /= len(test_models)

        if calc_metric:
            y_fitness = test_df['score'].values
            eval_results = scipy.stats.spearmanr(y_fitness, esb_ypred)[0]

        test_results = {}
        results_df = test_df.copy()
        results_df = results_df.drop(columns=['sequence'])
        results_df['prediction'] = esb_ypred
        test_results['df'] = results_df
        if save_prediction:
            self.saver.save_df(results_df, 'prediction.tsv')
        test_results['loss'] = esb_loss
        if calc_metric:
            test_results['metric'] = {'corr': eval_results}
        return test_results
