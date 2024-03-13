library(Tempora)
library(Seurat)
library(anndata)
library(RCurl)

# Pick dataset
adata <- read_h5ad("/data/COVID_gse212041_pca50.h5ad")
adata$obs['Clusters'] <- adata$obs['leiden']
adata$obs['Timepoints'] <- adata$obs['visit']

tempora_obj <- CreateTemporaObject(
  t(adata$X),
  meta.data=adata$obs,
  timepoint_order=c('D0', 'D3', 'D7')  # Set timepoint order for each dataset
)

gmt_url = "http://download.baderlab.org/EM_Genesets/current_release/Human/symbol/"

#list all the files on the server
filenames = getURL(gmt_url)
tc = textConnection(filenames)
contents = readLines(tc)
close(tc)

#get the gmt that has all the pathways and does not include terms inferred from electronic annotations(IEA)
#start with gmt file that has pathways only
rx = gregexpr("(?<=<a href=\")(.*.GOBP_AllPathways_no_GO_iea.*.)(.gmt)(?=\">)",
  contents, perl = TRUE)

gmt_file = unlist(regmatches(contents, rx))
dest_gmt_file <- file.path(getwd(),gmt_file )
download.file(
    paste(gmt_url,gmt_file,sep=""),
    destfile=dest_gmt_file
)

#Estimate pathway enrichment profiles of clusters
tempora_obj <- CalculatePWProfiles(tempora_obj, gmt_path = gmt_file,
                method="gsva", min.sz = 5, max.sz = 400, parallel.sz = 1)

#Build trajectory with 6 PCs
tempora_obj <- BuildTrajectory(tempora_obj, n_pcs = 6, difference_threshold = 0.01)

#Visualize the trajectory
tempora_obj <- PlotTrajectory(tempora_obj)

#Fit GAMs on pathway enrichment profile
tempora_obj <- IdentifyVaryingPWs(tempora_obj, pval_threshold = 0.05)

#Plot expression trends of significant time-varying pathways
PlotVaryingPWs(tempora_obj)
