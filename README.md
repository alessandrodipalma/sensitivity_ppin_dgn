# Sensitivity analysis on Protein-Protein Interaction Networks through Deep Graph Networks
Code repository of the work "Sensitivity analysis on Protein-Protein Interaction Networks through Deep Graph Networks", sumbitted for publication at [BMC Bioinformatics](https://bmcbioinformatics.biomedcentral.com/).
## Setup

As a first step, install the required packages
```
    pip install -r requirements.txt
```
Next, download and extract the [data repository](10.5281/zenodo.14535760) and ensure that the paths in `config.json` point to the directories where you extracted the data.


## Model training and evaluation

To train the model over the Biogrid PPI Network subgraphs corresponding to the input/output pairs in DyPPIN run:
```
    python -m src.model_development.training 
```

The default hyperparameters are set to the best configurations found through grid searches. Any hyperparameter can be set as command line argument, defined in `src/model_development/cli.py`.
When running with the `--production` flag, the script will use the whole dataset for the training, and produce the checkpoint used in the `prediction` module.

The model selection process is contatined in `notebooks/model_selection.ipynb`.
The code to produce the model analysis and the plots present in Figure 7 can be found in `notebooks/model_analysis.ipynb`.

## Predicting sensitivity over protein pairs

To predict sensitivity over any BioGRID PPIN subgraph, just run:
```
    python -m src.prediction.sensitivity_prediction.py --p <proteins.txt> --p-in <input_proteins.txt> --p-out <output_proteins.txt>
```
The script will automatically extrtact the proteins subgraph from the BioGRID database in tabular data, present in `data/external_data/biogrid`. From the script parameters, you can also select the specific PPIN file, so you can use any PPIN put in the same data format.
The script will print a coverage of the desired proteins w.r.t. DyPPIN, and an estimation of the confidence given the graph topology. We suggest to run the prediction over subgraphs with not much more nodes than the ones in the training data (40 nodes).

## Predicting sensitivity from datalist

If you already have extracted the graphs on which you want to perform the predictions, you can convert them from networks to the needed pyg format using the utility function `src.prediction.build_graphs.nx_to_pyg`. Once you have you datalist, run:

```
    python -m src.prediction.sensitivity_prediction_datalist --datalist {your_data_path}/data_list.pickle --gpu 0 --outpath {your_data_path}/predictions.tsv
```

## Case study (BACH2 relevance in insulin and glucagon regulation)

The notebook `case_study_bach2.ipynb` presents the analysis of the models prediction and the code to reproduce Figure 8.
