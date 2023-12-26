# LASSO: Machine Learning based Feature Selection using Normalized RNA-seq Counts
LASSO is statistical modeling and machine learning approach used to estimate the relationships between variables, make prediction and select the potential features. Here we report a python code named LASSO_PY; to select potential features such as lncRNAs or transcription factors (TFs) associated with trait using normalized RNA-seq counts.

# What we need ?
We needed normalized RNA-seq counts for TFs and lncRNAs or TFs and miRNAs in TSV matrix

# Getting started 
1. Once you are ready with your datasets, update your conda environment with necessary pacakges using following commamd
   ```
   mamba env update --file LASSO.yml
   ```
2. Download the LASSO_PY repository to your local system as follows
   ```
   git clone https://github.com/nikhilshinde0909/LASSO_PY.git
   ```
3. One softwares are installed and repository is cloned, task is ready to execute.
4. Use following command to understand inputs neede for code
   ```
   python path_to_LASSO_PY/bin/LASSO_feature_selection.py -h
   ```
5. To perform feature selection using lncRNAs and TFs matrix refer following command
   ```
    python path_to_LASSO_PY/bin/LASSO_feature_selection.py TF_matrix.TSV Lnc_matrix.TSV
   ```
6. You can use sample datasets given with the code


# Thanks for Using LASSO_PY
