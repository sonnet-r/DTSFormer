# DTSFormer: Decoupled Temporal-Spatial Diffusion Transformer

[![Paper](https://img.shields.io/badge/Paper-10.1016%2Fj.knosys.2024.112828-blue)](https://doi.org/10.1016/j.knosys.2024.112828)
[![Journal](https://img.shields.io/badge/Knowledge--Based%20Systems-Volume%20309-green)](https://doi.org/10.1016/j.knosys.2024.112828)
[![Framework](https://img.shields.io/badge/PyTorch-Official%20Implementation-ee4c2c)](https://pytorch.org/)

> **Official PyTorch implementation** of the Knowledge-Based Systems paper **"DTSFormer: Decoupled Temporal-Spatial Diffusion Transformer for Enhanced Long-Term Time Series Forecasting"**, Volume 309 (2025), Article 112828.

This repository contains the authors' official source code for DTSFormer, a decoupled temporal-spatial diffusion Transformer for multivariate long-term time series forecasting.

**Keywords:** DTSFormer, time series forecasting, long-term forecasting, multivariate time series, temporal-spatial Transformer, graph diffusion, seasonal-trend decomposition, Knowledge-Based Systems.

## 📰 News

- **2024-12-21:** The paper became available online in *Knowledge-Based Systems*. [[Paper]](https://doi.org/10.1016/j.knosys.2024.112828)
- **2024-11-29:** The paper was accepted by *Knowledge-Based Systems*.

## 🌟 Overview

DTSFormer separates temporal and spatial dependencies after seasonal-trend decomposition. Its main components are:

- **Temporal Information Fusion (TIF):** integrates local seasonal and trend information through learnable memory attention.
- **Adaptive Mix-hop Diffusion:** learns sparse seasonal and trend graphs and propagates information over multiple hops.
- **Cross-diffusion Attention:** iteratively exchanges information between the seasonal and trend spatial representations.
- **Decoupled Forecasting:** combines temporal and spatial forecasts to produce the final prediction.

The released model follows a single complete inference path. Every model component in `models/DTSFormer.py` contributes to the final forecast.

## 🛠 Prerequisites

The code is implemented with Python 3 and PyTorch. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## 📊 Prepare Datasets

The data loader supports ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Weather, Solar-Energy, Exchange-Rate, and custom multivariate datasets.

Download the public forecasting datasets from the dataset collection provided by [Time-Series-Library](https://github.com/thuml/Time-Series-Library), then place them under `./data/`:

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

CSV files must contain a `date` column. Solar-Energy uses the standard comma-separated text format. Dataset paths can also be specified with `--root_path` and `--data_path`.

## 💻 Training

Train and evaluate DTSFormer directly through `run.py`. For example, ETTh1 with an input length of 96 and a prediction length of 96 can be run as follows:

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

To use another supported dataset, change `--data`. Dataset-specific channel counts, batch sizes, learning rates, graph neighborhoods, and diffusion depths are configured automatically. Important settings can still be overridden from the command line:

```bash
python run.py \
  --is_training 1 \
  --model_id Traffic_96_96 \
  --data Traffic \
  --pred_len 96 \
  --d_model 64 \
  --d_ff 128 \
  --graph_top_k 20 \
  --gcn_depth 3 \
  --cross_diffusion_iters 3 \
  --use_amp
```

Checkpoints are selected only by validation loss. After training and early stopping, the selected checkpoint is evaluated once on the test set. Metrics are written to `outputs/`.

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

## 🙏 Acknowledgement

We appreciate the following repositories for their valuable code and dataset resources:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [MTGNN](https://github.com/nnzhan/MTGNN)

## 📩 Contact

For questions about the code, please open an issue in this repository.
