import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import scipy.stats
import pathlib
import copy
import glob
import os

from data_Ankh import Dataset
from A3_2 import Predictor
import torch.nn.functional as F
from utils import Saver, EarlyStopping, Logger
from sklearn.model_selection import KFold

class V2(object):
    def __init__(self,
                 output_dir=None,
                 train_tsv=None,
                 test_tsv=None,
                 fasta=None,
                 d_model=2560, d_h=128, d_attn=3,
                 save_log=False):
        self.dataset = Dataset(
            train_tsv=train_tsv, test_tsv=test_tsv, fasta=fasta)

        self.saver = Saver(output_dir=output_dir)
        self.logger = Logger(logfile=self.saver.save_dir / 'exp.log' if save_log else None)
        self.d_model = d_model
        self.d_h = d_h
        self.d_attn = d_attn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir

        self.criterion = F.mse_loss

    def train_kfold(self, n_splits=10, epochs=200, log_freq=10, patience=20,
                save_checkpoint=False, shuffle=True, random_state=42, resume_path=None):
        """
        K-fold CV training (e.g. 10% hold-out each fold when n_splits=10).
        """    
        full_df = self.dataset.return_full_df().reset_index(drop=True)
        n_samples = len(full_df)
        fold_results = []
        overall_predictions = np.zeros(n_samples, dtype=np.float32)
        overall_truth = full_df['score'].values.astype(np.float32)

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.logger.info(f"Starting {n_splits}-fold CV on {n_samples} samples")
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(full_df)):
            self.logger.info(f" Fold {fold_idx}/{n_splits}: train {len(train_idx)} hold {len(test_idx)}")
            
            train_df = full_df.iloc[train_idx].reset_index(drop=True)
            test_df  = full_df.iloc[test_idx].reset_index(drop=True)

            train_data = self.dataset.build_data(train_df, return_df=False)
            test_data = self.dataset.build_data(test_df,  return_df=False)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_df), shuffle=True)
            test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=1, shuffle=False)

            model     = Predictor(out_dim=self.d_model, d_h=self.d_h, d_attn=self.d_attn).to(self.device)
            optimizer = optim.Adam(model.parameters())
            
            if resume_path is not None:
                ckpt_path = os.path.join(resume_path, f"model_fold_{fold_idx+1}.pt")
                self.logger.info(f"Resuming fold {fold_idx+1} from {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location=self.device)
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                
            stopper   = EarlyStopping(patience=patience, eval_freq=log_freq)
            best_model_state_dict = None
            
            for epoch in range(1, epochs + 1):
                model.train()
                tot_loss = 0
                for step, batch in enumerate(train_loader, 1):
                    y = batch['label'].to(self.device)
                    X = batch['glob_feat'].to(self.device)
                    optimizer.zero_grad()
                    output = model(X).view(-1)
                    loss = self.criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    tot_loss += loss.item()

                if epoch % log_freq == 0:
                    model.eval()
                    tot_val_loss = 0
                    with torch.no_grad():
                        for batch in test_loader:
                            X = batch['glob_feat'].to(self.device)
                            output = model(X).view(-1)
                            y_true = batch['label'].to(self.device)
                            tot_val_loss += self.criterion(output, y_true).item()
                    val_loss = tot_val_loss / len(test_loader)
                    current_score = -val_loss  # using negative loss for early stopping
                    if stopper.update(current_score):
                        best_model_state_dict = copy.deepcopy(model.state_dict())
                    self.logger.info(
                        f"Fold {fold_idx+1}, Epoch {epoch}, Train Loss: {tot_loss/step:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Best Score: {stopper.best_score:.4f}"
                    )
                    if stopper.early_stop:
                        self.logger.info(f"Early stopping in fold {fold_idx+1} at epoch {epoch}")
                        break

            if best_model_state_dict is not None:
                model.load_state_dict(best_model_state_dict)

            # Save the model for this fold if required.
            if save_checkpoint:
                checkpoint_dir = self.output_dir + '/kfold'
                os.makedirs(self.output_dir, exist_ok=True)
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_path = os.path.join(checkpoint_dir, f"model_fold_{fold_idx + 1}.pt")
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # Optionally add training meta-data
                }
                torch.save(checkpoint, save_path)
                self.logger.info(f"Saved model for fold {fold_idx+1} to {save_path}")

            # Evaluate on the left‑out sample.
            test_results = self.test(test_model=model, test_loader=test_loader, test_df=test_df)
            fold_results.append(test_results)
            pred_value = test_results['df']['prediction']
            overall_predictions[test_idx]=pred_value
            self.logger.info(
                f"Fold {fold_idx+1} test: Loss: {test_results['loss']}"
            )
        overall_predictions = np.array(overall_predictions)
        overall_corr = scipy.stats.spearmanr(overall_truth, overall_predictions)[0]
        overall_loss = np.mean([res['loss'] for res in fold_results])
        self.logger.info(
            f"kfold Overall Performance - Avg Loss: {overall_loss:.4f}, "
            f"Spearman Correlation: {overall_corr:.4f}"
        )

        # Optionally, return the list of saved model paths for further use.
        return fold_results
            
    
    def test(self, test_model=None, test_loader=None, test_df=None,
                checkpoint_dir=None, save_prediction=False):
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
                for step, batch in tqdm(enumerate(test_loader, 128),
                        leave=False, total=len(test_loader)):
                    X = batch['glob_feat'].to(self.device)
                    output = model(X)
                    output = output.view(-1)
                    y = batch['label'].to(self.device)
                    loss = self.criterion(output, y)
                    tot_loss += loss.item()
                    y_pred = output if y_pred is None else torch.cat((y_pred, output), dim=0)

            y_pred = y_pred.detach().cpu() if self.device == torch.device('cuda') else y_pred.detach()
            esb_ypred = y_pred.view(-1, 1) if esb_ypred is None else torch.cat((esb_ypred, y_pred.view(-1, 1)), dim=1)
            esb_loss += tot_loss / step

        esb_ypred = esb_ypred.mean(axis=1).numpy()
        esb_loss /= len(test_models)

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
        test_results['metric'] = {'corr': eval_results}
        return test_results
    
    def ensemble_predict(self, model_paths):
        predictions = []
        esb_ypred = None
        test_loader, test_df = self.dataset.get_test_dataloader(batch_size=128, return_df=True)
        checkpoint_dir = pathlib.Path(model_paths)
        if not checkpoint_dir.is_dir():
            raise ValueError(f'{checkpoint_dir} is not a directory')
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        for checkpoint_path in checkpoint_files:
            print(f'Loaded checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path)
            model = Predictor(out_dim=self.d_model, d_h=self.d_h, d_attn=self.d_attn).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            y_pred = None
            with torch.no_grad():
                for step, batch in tqdm(enumerate(test_loader, 128),
                        leave=False, total=len(test_loader)):
                    X = batch['glob_feat'].to(self.device)
                    output = model(X)
                    output = output.view(-1)
                    output = output.detach().cpu()
                    y_pred = output if y_pred is None else torch.cat((y_pred, output), dim=0)
            y_pred = y_pred.detach().cpu() if self.device == torch.device('cuda') else y_pred.detach()
            esb_ypred = y_pred.view(-1, 1) if esb_ypred is None else torch.cat((esb_ypred, y_pred.view(-1, 1)), dim=1)
        # Average predictions across folds.
        esb_ypred = esb_ypred.mean(axis=1).numpy()
        results_df = test_df.copy()
        results_df = results_df.drop(columns=['sequence'])
        results_df['prediction'] = esb_ypred
        self.saver.save_df(results_df, 'prediction.tsv')
        return predictions

        
        
    
    


    
