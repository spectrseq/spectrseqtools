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


### Tags

For easier usage, we predefined a set of (5'- or 3'-) tags, which can be used via 
identifier (`ID`). 
Note that the set also contains natural ends (like OH, CP, and P):

| ID                 | 5'-Tag | 3' Tag | Name                                      | Mass      |
|--------------------|--------|--------|-------------------------------------------|-----------|
| OH                 | X      | X      | Oxygen (natural)                          | 18.009482 |
| CP                 |        | X      | Cyclic phosphate (natural)                | 79.965826 |
| P                  | X      | X      | Phosphate (natural)                       | 97.975308 |
| PS                 | X      |        | Thiophosphate                             | 113.95405 |
| THIOAMIDE          | X      |        | Thioamide                                 | 170.98    |
| IDT_6FAM           | X      |        | 6FAM (Integrated DNA Technologies Inc.)   | 555.1294  |
| BIOMERS_6FAM       | X      |        | 6FAM (Biomers.net GmbH)                   | 555.1294  |
| PCP_BIOTIN         |        | X      | pCp Biotin                                | 728.2     |
| IDT_BIOTIN         |        | X      | Biotin (Integrated DNA Technologies Inc.) | 455.1491  |
| BIOMERS_BIOTIN_TEG |        | X      | Biotin TEG (Biomers.net GmbH)             | 587.2278  |

Custom end tags can be defined via their mass. Note that the mass should assume a 
neutral state and contain the phosphate backbone.
