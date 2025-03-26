# zero-shot-das-denoising

This repository contains the implementation of my bachelor's thesis *Zero-Shot Denoising of Distributed Acoustic Sensing Data using Deep Priors*.

## About

*Distributed Acoustic Sensing* (DAS) is a novel technology that transforms standard fiber-optic cables into dense arrays of seismic sensors, enabling high-resolution vibration measurements over long distances.
However, DAS data is often contaminated with noise from various sources, making effective denoising crucial for practical use.

*Zero-shot* denoising refers to denoising methods which do not require any pre-collected training datasets.

*Deep Image Prior* (DIP) is such a method operating in a zero-shot setting.
It works by by optimizing a neural network on a single noisy instance, using the inherent structural bias of convolutional networks to favor signal over noise.

## Getting Started

To set up this project, please follow these instructions:

1. Clone this repository
2. Set up large file storage:
    - Install **git-lfs**
    - Run `git lfs install` and `git lfs pull`
<!-- TODO 3. Install dependencies: `pip install -f requirements.txt` -->

Now you are ready to explore the DIP-based methods discussed in my thesis.

Example notebooks for both image and DAS data are provided in the `code/examples` directory.
