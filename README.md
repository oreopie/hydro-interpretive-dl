## An interpretive deep learning approach to investigating flooding mechanisms
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.4686106-blue.svg)](https://doi.org/10.5281/zenodo.4686106)

- [Overview](#overview)
- [Quick Start](#quick-start)

### Overview

The code demonstrates how to interpret LSTM-based hydrological models through the use of expected gradients and additive decomposition methods, as described in paper

Jiang, S., Zheng, Y., Wang, C., & Babovic, V. (2021). **Uncovering flooding mechanisms across the contiguous United States through interpretive deep learning on representative catchments**. *Water Resources Research*, 57, e2021WR030185. https://doi.org/10.1029/2021WR030185

Please refer to the file [LICENSE](/LICENSE) for the license governing this code.

Kindly contact us with any questions or ideas you have concerning the code, or if you discover a bug. You may [raise an issue](https://github.com/oreopie/hydro-interpretive-dl/issues) here or contact Shijie Jiang through email at *jiangsj(at)mail.sustech.edu.cn*

------

### Quick Start

The code was tested with Python 3.7. To use the code, please do:

1. Clone the repo:

   ```shell
   git clone https://github.com/oreopie/hydro-interpretive-dl.git
   cd hydro-interpretive-dl
   ```

> The study is implemented based on MOPEX (Model Parameter Estimation Experiment) dataset by [*NOAA National Weather Service*](https://www.nws.noaa.gov/ohd/mopex/mo_datasets.htm). 
> One can download the data from the [official website](https://hydrology.nws.noaa.gov/pub/gcip/mopex/US_Data/) or [HydroShare](https://www.hydroshare.org/resource/99d5c1a238134ea6b8b767a65f440cb7/data/contents/MOPEX.zip) into `mopex`.

2. Install dependencies ([conda](https://docs.conda.io/en/latest/miniconda.html) is recommended to manage packages):

   ```shell
	conda create -n hydrodeepx
	conda activate hydrodeepx
	conda install -c conda-forge python=3.7 numpy=1.16.4 tensorflow=1.14 h5py=2.10 keras shap
	conda install -c conda-forge pandas scipy matplotlib jupyter tqdm
   ```
   
   Note for this implementation, `tensorflow v1.14` is recommended, though `tensorflow v2.x` may also work.

3. Start `Jupyter Notebook` and run the `interpret_lstm.ipynb` locally.
