from torch.utils.data import DataLoader

from data_provider.data_loader import MyDataset


def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
    else:
        shuffle_flag = True

    data_set = MyDataset(
        root_path=args.root_path,
        data_path=args.data_path,
        data=args.data,
        flag=flag,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        features=args.features,
        target=args.target,
        freq=args.freq
    )
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=True
    )

    return data_set, data_loader
