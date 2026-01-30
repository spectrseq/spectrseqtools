# SpectrSeqTools

SpectrSeqTools is a fully automatic analysis platform for sequencing small RNA molecules including 144 known post translational modifications measured via LC-MS/MS data.

## Installation

We recommend installing SpectrSeqTools via the pixi package manager:

```bash
pixi global install -c conda-forge -c bioconda spectrseqtools
```

Alternatively, a conda environment (for use with e.g. Snakemake) can de defined as follows

```yaml
channels:
  - conda-forge
  - bioconda
  - nodefaults
dependencies:
  - spectrseqtools
```
For reproducibility, make sure to specify the version you want to use next to the package name above.

## Usage

After installing, run

```bash
spectrseqtools --help
```

to get help using SpectrSeqTools.
More extensive documentation with examples will appear here soon.
