# DTSFormer

**📄 Paper:** *DTSFormer: Decoupled Temporal-Spatial Diffusion Transformer for Enhanced Long-Term Time Series Forecasting*

**🏛️ Status:** Published in *Knowledge-Based Systems*, Volume 309 (2025), Article 112828. [[Paper]](https://doi.org/10.1016/j.knosys.2024.112828)

Official PyTorch implementation of DTSFormer for multivariate long-term time series forecasting.

## ✨ Overview

DTSFormer explores decoupled temporal-spatial diffusion modeling after seasonal-trend decomposition. This repository provides the complete model, dataset loaders, and reproducible training and evaluation entry points.

## 🛠️ Requirements

The code is implemented with Python 3 and PyTorch. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## 📊 Datasets

The data loader supports ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Weather, Solar-Energy, Exchange-Rate, and custom multivariate datasets.

Download the public datasets from [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and place them under `./data/`:

```text
data/
|-- ETTh1.csv
|-- ETTh2.csv
|-- ETTm1.csv
|-- ETTm2.csv
|-- electricity.csv
|-- traffic.csv
|-- weather.csv
|-- solar_AL.txt
`-- exchange_rate.csv
```

CSV files must contain a `date` column. Solar-Energy uses the standard comma-separated text format. Paths can be changed with `--root_path` and `--data_path`.

## 🚀 Training

Train DTSFormer through `run.py`. The following example uses ETTh1 with a 96 -> 96 forecasting setting:

```bash
python run.py \
  --is_training 1 \
  --model_id ETTh1_96_96 \
  --data ETTh1 \
  --root_path ./data/ \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96
```

Dataset-specific channel counts, batch sizes, learning rates, graph neighborhoods, and diffusion depths are configured automatically and can be overridden from the command line. Checkpoints are selected only by validation loss; the selected checkpoint is evaluated once on the test set.

## 📈 Evaluation

Evaluate an existing checkpoint with the same experiment configuration:

```bash
python run.py \
  --is_training 0 \
  --model_id ETTh1_96_96 \
  --data ETTh1 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96
```

Use `--save_predictions` to additionally save `pred.npy` and `true.npy`.

## 📚 Citation

If this work is helpful to your research, please consider citing:

```bibtex
@article{zhu2025dtsformer,
  title={DTSFormer: Decoupled Temporal-Spatial Diffusion Transformer for Enhanced Long-Term Time Series Forecasting},
  author={Zhu, Jiaming and Liu, Dezhi and Chen, Huayou and Liu, Jinpei and Tao, Zhifu},
  journal={Knowledge-Based Systems},
  volume={309},
  pages={112828},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.knosys.2024.112828}
}
```

## 🙏 Acknowledgements

We appreciate the following repositories for their valuable code and dataset resources:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [MTGNN](https://github.com/nnzhan/MTGNN)

## 📬 Contact

For questions about the code, please open an issue in this repository.
