import anndata
import pandas as pd


def adata_from_meta(
    d_path: str,
    meta_path: str,
    right_on: str,
    sep: str = '\t',
    index_col: int | None = 0,
    **kwargs,
) -> anndata.AnnData:
    """Reads a tsv file of RNA counts and a tsv file of metadata and
    combines them into an AnnData object.
    """
    df = pd.read_csv(d_path, sep=sep, index_col=index_col, **kwargs)
    meta = pd.read_csv(meta_path, sep=sep, index_col=index_col, **kwargs)
    adata = anndata.AnnData(df.T)
    adata.obs = adata.obs.merge(meta, left_index=True, right_on=right_on)
    return adata
