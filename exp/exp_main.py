import os
import time
import warnings

import numpy as np
import torch
from torch import optim, nn
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Linear, Linear_BatchNorm, Linear_MinusLast, Linear_RevIN, Linear_DishTS
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Linear': Linear,
            'Linear_BatchNorm': Linear_BatchNorm,
            'Linear_MinusLast': Linear_MinusLast,
            'Linear_RevIN': Linear_RevIN,
            'Linear_DishTS': Linear_DishTS,

        }
        model = model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, criterion):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        outputs = self.model(batch_x, batch_x_mark, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, :, f_dim:]
        batch_y = batch_y[:, :, f_dim:]

        loss = criterion(outputs, batch_y)
        return loss, outputs, batch_y

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(vali_loader, desc='vali: '):
                loss, _, _ = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, criterion)
                total_loss.append(loss.cpu())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.save_path + '/checkpoints/', setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(train_loader, desc='train: '):
                model_optim.zero_grad()
                loss, _, _ = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, criterion)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0} cost time: {1}".format(epoch + 1, time.time() - epoch_time))
            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        print('loading model')
        criterion = self._select_criterion()
        self.model.load_state_dict(
            torch.load(os.path.join(self.args.save_path + '/checkpoints/' + setting, 'checkpoint.pth'))
        )

        self.model.eval()
        with torch.no_grad():

            if setting.split('_')[-1] == '0':
                folder_path = self.args.save_path + '/test_results/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

            preds, trues = [], []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader, desc='test: ')):
                _, outputs, batch_y = self._process_one_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, criterion)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)

                if i % 20 == 0 and setting.split('_')[-1] == '0':
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print('Test shape:', preds.shape)

            if setting.split('_')[-1] == '0':
                folder_path = self.args.save_path + '/results/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print('mse:{}, mae:{}'.format(mse, mae))
            f = open("result.txt", 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

            if setting.split('_')[-1] == '0':
                np.save(folder_path + 'metrics.npy', np.array([mae, mse]))
                np.save(folder_path + 'pred.npy', preds)
                np.save(folder_path + 'true.npy', trues)

        return mse, mae
