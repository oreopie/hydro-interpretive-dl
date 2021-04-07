## Interpretive deep learning for hydrological understanding

- [Overview](#overview)
- [Quick Start](#quick-start)

### Overview
The code demonstrates the implementation of the interpretive LSTM proposed in paper "***Gaining process understanding from the black box: Uncovering flooding mechanisms through interpretive deep learning***" (submitted to a journal)

Please refer to the file [LICENSE](/LICENSE) for the license governing this code.

If you have any questions or suggestions with the code or find a bug, please let us know. You are welcome to [raise an issue here](https://github.com/oreopie/hydro-interpretive-dl/issues) or contact Shijie Jiang at *jiangsj(at)mail.sustech.edu.cn*

------

### Quick Start

The code was tested with Python 3.7. To use the code, please do:

1. Clone the repo:

   ```shell
   git clone https://github.com/oreopie/hydro-interpretive-dl.git
   cd hydro-interpretive-dl
   ```

2. Install dependencies ([conda](https://docs.conda.io/en/latest/miniconda.html) is recommended to manage packages):

   ```shell
	conda create -n hydrodeepx
	conda activate hydrodeepx
	conda install -c conda-forge python=3.7 numpy=1.16.4 pandas scipy tensorflow=1.14 matplotlib jupyter h5py=2.10 shap tqdm
   ```

3. Start `Jupyter Notebook` and run the `demo.ipynb` locally.

> The study is implemented based on MOPEX (Model Parameter Estimation Experiment) dataset by [*NOAA National Weather Service*](https://www.nws.noaa.gov/ohd/mopex/mo_datasets.htm). One can download the data from the [official website](https://hydrology.nws.noaa.gov/pub/gcip/mopex/US_Data/) or [HydroShare](https://www.hydroshare.org/resource/99d5c1a238134ea6b8b767a65f440cb7/data/contents/MOPEX.zip) into `mopex`.
