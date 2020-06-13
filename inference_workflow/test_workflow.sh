#!/bin/bash
#SBATCH --mem=10gb
#SBATCH --ntasks=5

luigid --background
PYTHONPATH='.' luigi --module inference_workflow Strain_Finder --workers 5
