# Skin Lesion Characterization using Conditional GANs and EfficientNet

Analysis of **Skin Lesion Images** to segment lesion regions and characterize the lesion type using deep adversarial learning.

## Quick links

- [Manuscript draft](./assets/manuscript-draft.pdf).
- [Data directory organization](./datasets/README.md) (internal reference).

## Environment setup

Set up the execution environment using the requirement files.
- Requirements for setting up **conda environment** are contained in `dep-file-conda.yml`.
- Requirements for setting up using **pip installations** (not recommended) are contained in `dep-file-pip.txt`.

## Running experiments for dataset balancing and lesion segmentation

The dataset balancing analysis and lesion segmentation network is contained in [`notebooks`](./notebooks) as sequentially numbered python notebooks.

## Running scaling experiments for classification

The classification architectures for classifiers are scripted in [`src/classifiers`](./src/classifiers). The preprocessing workflow used to prepare the dataset is in [`model-building`](.//model-building).

- Load the appropriate classifier drive function, say `experiment_effnetb6` in [`src/run.py`](./src/run.py) by importing them.
- Set up the data path.
- Call the driver function in [`src/run.py`](./src/run.py), and execute `python run.py`.

> [!Note]
> The dataset will be released during manuscript publication.
