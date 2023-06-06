# FDA

WWW 2023 Conference

Paper title: Improving Recommendation Fairness via Data Augmentation [arxiv](https://arxiv.org/abs/2302.06333) [WWW](https://dl.acm.org/doi/abs/10.1145/3543507.3583341)



## Prerequisites

- PyTorch
- Python 3.5
- NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Clone this repo

### Train/test


- Train FDA_BPR on MovieLens:

```bash
cd FDA_anonymous/fda_bpr_ml
python main.py
```

- Train FDA_GCCF on MovieLens:

```bash
cd FDA_anonymous/fda_gccf_ml
python main.py
```

- Train FDA_BPR on LastFM:

```bash
cd FDA_anonymous/fda_bpr_lastfm
python main.py
```

- Train FDA_GCCF on LastFM:

```bash
cd FDA_anonymous/fda_gccf_lastfm
python main.py
```

**Note**: The results of FDA will be output on the terminal after the training.
