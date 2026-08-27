from torch.utils.data import DataLoader

from data_provider.data_loader import (
    ETTHourDataset,
    ETTMinuteDataset,
    ForecastDataset,
    SolarDataset,
)


DATASETS = {
    "ETTh1": ETTHourDataset,
    "ETTh2": ETTHourDataset,
    "ETTm1": ETTMinuteDataset,
    "ETTm2": ETTMinuteDataset,
    "Weather": ForecastDataset,
    "Electricity": ForecastDataset,
    "Traffic": ForecastDataset,
    "Exchange": ForecastDataset,
    "Solar": SolarDataset,
    "Custom": ForecastDataset,
}


def data_provider(args, flag):
    dataset_class = DATASETS[args.data]
    dataset = dataset_class(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=(args.seq_len, args.label_len, args.pred_len),
        features=args.features,
        target=args.target,
        cycle_length=args.cycle_length,
    )
    if dataset.data_x.shape[1] != args.enc_in:
        raise ValueError(
            f"--enc_in is {args.enc_in}, but {args.data_path} contains "
            f"{dataset.data_x.shape[1]} selected feature columns"
        )
    print(f"{flag}: {len(dataset)} samples")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=flag == "train",
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=getattr(args, "use_gpu", False),
    )
    return dataset, loader
