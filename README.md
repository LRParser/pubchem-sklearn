# PubChem-sklearn

## A project to apply machine learning techniques to bioassay data hosted at PubChem

This is a project to analyze bioassay data from the PubChem service via various machine learning algorithms. Of particular interest is the large size of these bioassays (for the example assay here, over 220,000 compounds were tested)

### Background on the assay and terminology

The bioassay and relevant SMILES mappings were downloaded from the PubChem site directly for the sample bioassay 1030; this bioassay looks for inhibitors of the protein encoding gene ALDH1A1; this gene is associated with metabolic diseases.The bioassay tests a number of compounds; the SMILES mappings for these compounds was also downloaded from PubChem. The SMILES data is normalized in the form of Morgan Fingerprints which are then used to train various algorithms via sklearn. Results can be found [here](capstone_project_jh.md) and in PDF form [here](report.pdfW)

### Setup

Ubuntu 16.04, Python 3.6 as well as Anaconda 4.4.0 is required. The RDKit library is also required (installed via the below steps). You can setup a environment via either installing each conda dependency sequentially, or useing an environment file


Install steps:

```
conda create -n pubchem-sklearn anaconda
source activate pubchem-sklearn
./install_reqs.sh
```

### Usage

Load in the browser and run all cells of the `pubchem_bioassay_sklearn.ipynb` file via the command `jupyter notebook`

Note: You may require approx 64 GB of RAM to test the 

Run this file, then the `pubchem_dnn.ipynb` as well as the `pubchem_exploratory_visualization.ipynb` files.
