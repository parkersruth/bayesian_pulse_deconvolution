
This repository contains the code required to replicate the results from the following publication.

> P. S. Ruth, T. DeBenedetti, L. O’Brien, J. A. Landay, T. P. Coleman, and E. B. Fox, “Vascular waveform analysis using Bayesian pulse deconvolution,” npj Digit. Med., Sep. 2026, doi: [10.1038/s41746-026-03213-0](https://doi.org/10.1038/s41746-026-03213-0).


This code has been tested on MacOS.

## Code installation

1. Install the [(mini)conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [mamba](https://mamba.readthedocs.io/en/stable/installation/mamba-installation.html) Python environment manager.

2. Open a terminal and navigate to an empty working directory where you will download the code and data. Clone the GitHub repository and enter it:

3. Initialize the conda environments for the repository:

```
cd [path to working directory]
git clone https://github.com/parkersruth/bayesian_pulse_deconvolution.git
cd bayesian_pulse_deconvolution
conda env create --file envs/env_pulse_deconv.yml
conda env create --file envs/env_pyppg.yml
conda activate pulse_deconv
```

## Using Snakemake

The evaluation pipeline uses [Snakemake](https://snakemake.github.io/). This is a Makefile-like tool that automates parallel processing and intermediate file management. At any time, the Snakemake pipelines can be halted by pressing `CTL-C`. When restarted, Snakemake will resume from where it left off.

For convenience, we provide a `run.sh` script that pre-loads a base Snakemake shell command. To use it, you must first set execution privileges.

```
chmod u+x run.sh
```

You can then run the program like this:
```
./run.sh
```

By default, this command reproduces all analyses and figures managed with Snakemake. This may take multiple hours to run on a laptop computer. Analysis for individual figures can be run separately. For example:

```
./run.sh fig1
```
computes only the analyses to reproduce Figure 1.

Any additional arguments to the `run.sh` script are passed to the `snakemake` command. Some useful Snakemake commands are listed below:

- Use `-c[n]` to parallelize across n CPU cores, e.g. `-c4` parallelizes over 4 cores.
- Use `-n` to show a dry-run of the steps that will be run without actually running the jobs.
- Use `-k` to continue running other steps if an error occurs in one of them.
- Use `-F` to re-run all steps and over-write temporary files already generated.


## Simulation experiments

### Fig. 1, 2, 3

Run the simulations:
```
./run.sh fig1 fig2 fig3
```

To produce the figures, open and run `figs/fig1.ipynb`, `figs/fig2.ipynb`, and `figs/fig3.ipynb` in an IDE like VSCode, or from the command line like this:
```
cd figs/fig1
jupyter lab fig1.ipynb
```

### Fig. 5

Fig. 5 does not use Snakemake.

Compute the analytical solution to the sample waveform by running `figs/fig5/solve_rarker.ipynb`. This should generate a solution file `figs/fig5/rarker_bpd.pkl`.

Run the `figs/fig5/fig5_mcmc.ipynb` notebook to generate the MCMC posterior samples. If the kernel crashes, restart and run again. The script only re-runs parts of the code that have yet been saved to intermediate files.

Next, generate the figure by running the `figs/fig5/fig5.ipynb` notebook.


## Empirical experiments

Empirical experiments require access to the Aurora-BP dataset. Sample data and instructions for accessing the full dataset are available here: https://github.com/microsoft/aurorabp-sample-data/tree/main

The code expects to find the Aurora-BP dataset located in `../data/aurora` relative to the `bayesian_pulse_deconvolution` repository root. If it is in a different directory, update `Snakefile` to point to the correct path.


### Fig. 4

Load the Aurora-BP dataset and apply ECG signal pre-filtering by running `notebooks/aurora_data_load.ipynb`.

Generate training and testing data splits by running `notebooks/aurora_data_split.ipynb`.

Run the experiments:
```
./run.sh fig4
```

Generate the figure by running `figs/fig4/fig4.ipynb`.



## Supplementary figures


### Supplementary Fig. 1

Generate a dataframe of simulation conditions by running `figs/fig_trends/fig_trends_make_df.ipynb`.

Run the simulations:
```
./run.sh fig_trends
```

Generate the figure by running `figs/fig_trends/fig_trends.ipynb`.


### Supplementary Fig. 2

The covariate analysis uses the same dataframe produced for Fig. 4.

Run the experiments:
```
./run.sh fig4
```

Generate the figure by running `figs/fig_covariates/fig_covariates.ipynb`.

### Supplementary Figs. 3-5

Run the experiments:
```
./run.sh fig_relaxations fig_ablations
```

Generate the figures by running the files `figs/fig_relaxations/fig_relaxations.ipynb`, `figs/fig_ablations/fig_ablations.ipynb`, and `figs/fig_relaxations/fig_relaxed_solns.ipynb`.


### Re-generate data-driven priors

Parameters of data-driven priors are stored in the `prior_params` directory.

To re-generate pulse timing priors, run `notebooks/aurora_prior_z.ipynb`.

To re-generate pulse shape priors, run `notebooks/aurora_prior_h.ipynb`.


## Citing This Work

Please cite the following paper:

> P. S. Ruth, T. DeBenedetti, L. O’Brien, J. A. Landay, T. P. Coleman, and E. B. Fox, “Vascular waveform analysis using Bayesian pulse deconvolution,” npj Digit. Med., Sep. 2026, doi: [10.1038/s41746-026-03213-0](https://doi.org/10.1038/s41746-026-03213-0).

