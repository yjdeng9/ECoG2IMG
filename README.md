# ECoG2IMG
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

Welcome to ``ECoG2IMG``,  a novel pipeline as the first visual perception image reconstruction method based on human ECoG, which allows the restoration of details at high resolution. 


# Contents
- [Overview](#overview)
- [System Requirements](#System-Requirements)
- [Usage Guide](#usage-guide)
- [Demo](#Demo)
- [License](#license)
- [Citation](#Citation)


# Overview
ECoG2IMG consists of three main components:
  (1) A pre-trained ECoG encoder, Talairach coordinate alignment mask autoencoder (TA-MAE);
  (2) A representation aligner based on the probabilistic denoising diffusion model (DDPM);
  (3) An image reconstruction module that generates images from the low-dimensional representation of ECoG based on the trained model to transform brain signals into visual images.
![图片1](https://github.com/user-attachments/assets/0aa1bab1-d012-4c9e-b207-136ea693652c)



# System Requirements

## Hardware Requirements
- **Standard Computer:** A standard computer with sufficient RAM to support in-memory operations.
- **GPU:** A GPU with more than 16GB of memory is required to handle large-scale computations.

## Software Requirements

The ``ECoG2IMG`` development version has been tested on CentOS 7 but is also compatible with Windows environments. It is essential to ensure that the ``Pytorch`` environment and ``CUDA`` environment are properly installed. 


# Usage Guide
   
You can run ``ECoG2IMG`` using the following command:

```bash
bash script/pipline.sh
```


# Demo

The demo data has been uploaded. The full data and full demonstration process used to build the model can be found on CodeOcean when the work is published.


# License

This project is licensed under the Apache License, Version 2.0 and is open for any academic use. Papers related to this project will be submitted, please cite for use and contact the author for data acquisition.

**Yongjie Deng - dengyj9@mail2.sysu.edu.cn**



# Citation

The paper is currently under review.

For usage of the package and associated manuscript, please cite:
    **Reconstruction Visual Perceptual Representations from Human Intracranial Electrocorticography Signals.** Yongjie Deng and et al.

