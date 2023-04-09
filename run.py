import argparse
import random

import numpy as np
import torch

from exp.exp_main import Exp_Main

fix_seed = 42
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Long Time Series Forecasting Library')

# basic
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--model', type=str, default='Linear_RevIN',
                    help='[Linear, Linear_BatchNorm, Linear_MinusLast, Linear_RevIN]')

# data
parser.add_argument('--root_path', type=str, default='./data/ETT-small')
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--data', type=str, default='ETTh1')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--freq', type=str, default='T')

# forecasting
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--pred_len', type=int, default=96)

# model
parser.add_argument('--enc_in', type=int, default=7)

# optimization
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--itr', type=int, default=5)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--lradj', type=str, default='type1')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)

Exp = Exp_Main

mse_list, mae_list = [], []
for ii in range(args.itr):
    setting = '{}_{}_ft{}_sl{}_pl{}_{}'.format(
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        ii
    )

    exp = Exp(args)

    if args.is_training:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mse, mae = exp.test(setting)
    mse_list.append(mse)
    mae_list.append(mae)

    torch.cuda.empty_cache()
print('>>>>>>>>>>  {}  <<<<<<<<<<'.format(setting[:-2]))
print('MSE || Mean: {0:.4f} | Std : {1:.4f}'.format(np.mean(mse_list), np.std(mse_list)))
print('MAE || Mean: {0:.4f} | Std : {1:.4f}'.format(np.mean(mae_list), np.std(mae_list)))
print('>>>>>>>>>>  {}  <<<<<<<<<<'.format(setting[:-2]))

f = open("result.txt", 'a')
f.write('>>> {} <<<'.format(setting[:-2]) + "  \n")
f.write('MSE || Mean: {0:.4f} | Std : {1:.4f}'.format(np.mean(mse_list), np.std(mse_list)) + "  \n")
f.write('MAE || Mean: {0:.4f} | Std : {1:.4f}'.format(np.mean(mae_list), np.std(mae_list)) + "  \n")
f.write('>>> {} <<<'.format(setting[:-2]))
f.write('\n')
f.write('\n')
f.close()
