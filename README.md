# Variance deficit in internal–external AUC differences

Data and analysis code for a meta-research study of published imaging prediction models that report both an internal and an external validation AUC with 95% confidence intervals.

## Contents

- `data/radiomics/`, `data/dl/`: PubMed search pools (`pool_pmids.json`: query, search date, PMIDs), AUC values extracted from abstracts (`auc_values.csv`), study-level internal–external pairs (`pairs.csv`), pairs whose intervals were taken from PMC full text (`pairs_ft.csv`, `pairs_ft_stat.json`) and cohort sizes (`pairs_with_n.csv`).
- `data/radiomics_FT/`, `data/dl_FT/`: abstract and full-text pairs combined; these are the analysis samples (radiomics k = 181, deep learning k = 66, rows with `headline == True` and `comp == "int_ext"`).
- `code/`: scripts numbered in the order they are run.
- `results/`: outputs of steps 05–13.

Each row of the extracted data carries the PMID and the source sentence, so every value can be checked against the original article. Abstracts and full texts are not redistributed; steps 01, 04 and 06 download them from PubMed and PMC.

## Columns of `pairs.csv`

`pmid`; `sent` (sentence index in the abstract); `model_cls` (clinical, imaging, combined); `comp` (`int_ext` internal vs external, `train_ext` training vs external); `auc_a`, `lo_a`, `hi_a` (internal or training AUC and 95% CI); `auc_ext`, `lo_ext`, `hi_ext` (external AUC and 95% CI); `n_ext_in_sent`; `model_txt`; `headline` (main model of the study); `source` (abstract or fulltext).

## Running

Python 3.12; `pip install -r requirements.txt`. Run from this directory.

```
# extraction (network access to NCBI E-utilities required)
python -I code/01_fetch_abstracts.py
python -I code/02_extract_auc.py
python -I code/03_build_pairs.py
python -I code/04_fulltext_ci.py
python -I code/06_ci_vs_n.py
# deep learning sample: set DATA_DIR=data/dl POOL_FILE=data/dl/pool_pmids.json RESULT_TAG=_DL for the same steps

# analysis (offline, uses the files in data/)
DATA_DIR=data/radiomics_FT RESULT_TAG=_FT python -I code/05_dispersion.py
DATA_DIR=data/dl_FT RESULT_TAG=_DL_FT python -I code/05_dispersion.py
python -I code/07_selected_vs_comparator.py
python -I code/08_variance_ratio.py
python -I code/09_expected_r_high.py
python -I code/10_sensitivity.py
python -I code/11_threshold_sorting_symmetry.py
python -I code/12_selection_simulation.py
python -I code/13_dl_vs_radiomics.py
```

All resampling and simulation use seed 42. Re-running the extraction steps can give slightly different pools if PubMed records have changed since the search date.

## Licence

Code (`code/`): MIT licence (`LICENSE`). Data (`data/`, `results/`): CC BY 4.0 (`data/LICENSE.md`).
