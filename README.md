
## Interpretive deep learning for identifying flooding mechanisms
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.4686106-blue.svg)](https://doi.org/10.5281/zenodo.4686106)

- [Overview](#overview)
- [Quick Start](#quick-start)

### Overview

The repository contains codes that demonstrate the use of interpretation techniques to gain insights into flooding mechanisms from LSTM-based hydrological models, as described in the papers

> Jiang, S., Zheng, Y., Wang, C., & Babovic, V. (2022a). **Uncovering flooding mechanisms across the contiguous United States through interpretive deep learning on representative catchments**. *Water Resources Research*, 57, e2021WR030185. https://doi.org/10.1029/2021WR030185

and 

> Jiang, S., Bevacqua, E., & Zscheischler, J. (2022b). **River flooding mechanisms and their changes in Europe revealed by explainable machine learning**, *Hydrology and Earth System Sciences*, 26, 6339–6359, https://doi.org/10.5194/hess-26-6339-2022

Please refer to the file [LICENSE](/LICENSE) for the license governing this code.

Kindly contact us with any questions or ideas you have concerning the code, or if you discover a bug. You may [raise an issue](https://github.com/oreopie/hydro-interpretive-dl/issues) here or contact Shijie Jiang through email at *shijie.jiang(at)hotmail.com*

------

### Quick Start

The code was tested with Python 3.7. To use the code, please do:

1. Clone the repository:

   ```shell
   git clone https://github.com/oreopie/hydro-interpretive-dl.git
   cd hydro-interpretive-dl
   ```

> The study (Jiang et al., 2022a) was implemented based on MOPEX (Model Parameter Estimation Experiment) dataset by [*NOAA National Weather Service*](https://www.nws.noaa.gov/ohd/mopex/mo_datasets.htm). One can download the data from the [official website](https://hydrology.nws.noaa.gov/pub/gcip/mopex/US_Data/) or [HydroShare](https://www.hydroshare.org/resource/99d5c1a238134ea6b8b767a65f440cb7/data/contents/MOPEX.zip) into `mopex`.

> The study (Jiang et al., 2022b) was implemented based on the following datasets:
> - GRDC dataset (https://www.bafg.de/GRDC)
> - E-OBS gridded precipitation and temperature dataset (https://www.ecad.eu/download/ensembles/download.php)
> - Catchment attributes and boundaries obtained from the Global Streamflow Indices and Metadata Archive (GSIM) (https://doi.pangaea.de/10.1594/PANGAEA.887477) and GRDC (https://www.bafg.de/GRDC/EN/02_srvcs/22_gslrs/222_WSB/watershedBoundaries.html)
> 
> We provide a dataset for a sample catchment that contains daily precipitation, temperature, and discharge in `data`.

2. Install dependencies ([conda](https://docs.conda.io/en/latest/miniconda.html) is recommended to manage packages):

   ```shell
	conda create -n hydrodeepx python=3.6.6
	conda activate hydrodeepx
	pip install innvestigate  tensorflow==1.13.1 h5py==2.10 numpy keras shap  matplotlib jupyter tqdm
   ```
   
   Note for this implementation, `tensorflow v1.x` is recommended, though `tensorflow v2.x` may also work.

3. Start `Jupyter Notebook` and run the Jupyter Notebooks in the repository locally.
