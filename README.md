# Strain-Detangler: Inferring Strains from Metagenomics Samples

## Setting up the environment:

Fork and clone strain-detangler.

Set up the conda environment appropriately, run `conda env create -f strains.yml`

## Installing Strain-Detangler:

To install, run `python setup.py install`

To edit and test in real-time, set up strain-detangler without installing by using `develop` instead of `install`

## Testing functions and workflows

### Strain-Detangler CLI and API

To test the core inference tools, run `python -m pytest`

### Workflows

Adjust the paths in the configuration file luigi.cfg, which can be found in the corresponding workflow directory. 

For example, to test the inference workflow, run `PYTHONPATH='.' luigi --module inference_workflow Strain_Finder --local`
PYTHONPATH is the location of the Luigi workflow Python code. If testing from a directory other than the workflow directory, set the PYTHONPATH as the workflow directory.
The `--local` flag indicates that all functions will be run by a single processor and no parallelization is used (i.e.: the Luigi daemon is not necessary). 
To test the workflow on a compute cluster with access to more processors, replace `--local` with `--workers=number_threads`. See the script `test_workflow.sh` for an example. 

To test the gold-standard file generator workflow, run `PYTHONPATH='.' luigi --module processor_evaluator_workflow Strain_GSTD_Processor --local`
To test the inference accuracy evaluator workflow, run `PYTHONPATH='.' luigi --module processor_evaluator_workflow Strain_Evaluator --local`
