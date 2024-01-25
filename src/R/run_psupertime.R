library(Seurat)
library(anndata)
library(psupertime)
library(yaml)
library(SingleCellExperiment)

# Pick dataset
adata <- read_h5ad("/data/COVID_gse212041_grinch_pca50.h5ad")
obj <- SingleCellExperiment(assays=list(logcounts=t(adata$X)), metadata = adata$obs)

psuper_obj <- psupertime(obj, obj@metadata$timepoint, lambdas=0.001)

psupertime_plot_all(psuper_obj, '/data/psuper/plots', label_name='Day', ext='pdf')

write_yaml(psuper_obj$ps_params, '/data/psuper/results/ps_params.yaml')
write.table(psuper_obj$beta_dt, '/data/psuper/results/beta_dt.csv')
write.table(psuper_obj$x_data, '/data/psuper/results/x_data.csv')
write.table(psuper_obj$proj_dt, '/data/psuper/results/proj_dt.csv')