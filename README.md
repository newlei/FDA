# FDA_anonymous
Used to anonymously share the FDA code for the WWW conference.

Paper title: Improving Recommendation Fairness via Data Augmentation

## Prerequisites

- PyTorch
- Python 3.5
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation

- Clone this repo:

```bash
git clone https://github.com/newlei/LR-GCCF.git
```

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
