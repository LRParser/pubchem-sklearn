# PubChem-sklearn

## A project to apply machine learning techniques to bioassay data hosted at PubChem

This is a project to analyze bioassay data from the PubChem service via various machine learning algorithms. Of particular interest is the large size of these bioassays (for the example assay here, over 220,000 compounds were tested)

### Background on the assay and terminology

The bioassay and relevant SMILES mappings were downloaded from the PubChem site directly for the sample bioassay 1030; this bioassay looks for inhibitors of the protein encoding gene ALDH1A1; this gene is associated with metabolic diseases.The bioassay tests a number of compounds; the SMILES mappings for these compounds was also downloaded from PubChem. The SMILES data is normalized in the form of Morgan Fingerprints which are then used to train various algorithms via sklearn. The first algorithm trained was RandomForestClassifier; the performance on a held-out test set was approximately 92%.

### Setup

Python 3.6 as well as Anaconda 4.4.0 is required. The RDKit library is also required (installed via the below steps). You can setup a environment via either installing each conda dependency sequentially, or useing an environment file

To install via environment file:

```
conda env create -f environment.yml
source activate pubchem-sklearn
```

Or to install manually:

```
conda create -n pubchem-sklearn anaconda
source activate pubchem-sklearn
conda install -y -q -c rdkit rdkit=2017.03.3
conda install nb_conda
```

### Usage

Load the `pcba_analysis.ipynb` file via the command `jupyter notebook`

After running all cells you will see the prediction accuracy of approximately 92% for the held-out test set on the sample bioassay 1030 (pubchem AID 1030)
