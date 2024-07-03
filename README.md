# Integrating patients in time series clinical transcriptomics data

This repository contains the source code for **Truffle** (Trajectory
Inference via Multi-commodity Flow with Node Constraints).

Processed and clustered data can be downloaded from [Google
Drive](https://drive.google.com/drive/folders/1WteYg1AXpsx_T7hfr5HKeCqQ23dCW88i?usp=share_link).
Place these files inside the `data` folder. To reduce and cluster the data,
we used the [`grinch`](https://github.com/euxhenh/grinch) library. The
config files used for `grinch` can be found in the `conf` folder.

To run Truffle, see the notebook `notebooks/multi-commodity_flow.ipynb`.

### Code overview

- `pyomo` implementation of multi-commodity flow is under `src/mc_flow.py`
- Truffle's main class is under `src/truffle.py`
- R scripts used to run Tempora and psupertime can be found in `src/R`


If you wish to perform your own clustering, update the `.obs['leiden']` key
in the AnnData object. Patient visits are under `.obs['visit']`. If you
wish to use `grinch` for clustering, run the following from the root of
Truffle's directory:

```bash
python /path-to-grinch-folder/src/grinch/main.py conf/config-file-you-wish-to-use.yaml
```

This will start the `grinch` pipeline and apply all the steps specified in
the config. Feel free to update parameters.