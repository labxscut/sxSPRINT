# Data

This repository does not include individual-level UK Biobank genotype, phenotype, PRS, or prediction files.

## Data Access

The study used UK Biobank data under approved application number 91090. Users should apply for UK Biobank access and generate their own disease-specific input matrices.

## Expected Input Files

For each disease-specific prediction task, SPRINT expects:

- `snp.csv`: a SNP dosage matrix with shape `(n_samples, n_snps)`;
- `prs.csv`: a standardized multi-trait PRS matrix with shape `(n_samples, n_prs)`;
- `labels.csv`: a binary case-control label vector with shape `(n_samples,)`.

Each CSV file may include a `sample_id` column. If `sample_id` is present, samples are aligned by ID across SNP, PRS, and label files. If `sample_id` is absent, rows are assumed to be in the same order.

Example:

```text
snp.csv
sample_id,rs1,rs2,rs3
S0001,0,1,2
S0002,2,0,1

prs.csv
sample_id,PRS_1,PRS_2
S0001,0.31,-1.12
S0002,-0.44,0.85

labels.csv
sample_id,label
S0001,0
S0002,1
```

SNP dosage values should be encoded as:

- `0`: no risk allele;
- `1`: one risk allele;
- `2`: two risk alleles.

PRS features should be standardized before model training. The same multi-trait PRS feature library can be reused across disease tasks, while the disease label and target-disease SNP set are disease-specific.

To train on another disease, place the corresponding disease-specific SNP matrix and label file in `data/`, update `configs/example_config.yaml`, and rerun `scripts/train_sprint.py`.

## Files Not Provided

The following file types should not be uploaded to GitHub:

- raw or processed UK Biobank genotype matrices;
- sample-level prediction files;
- files containing participant identifiers;
- large `.pkl`, `.npy`, or `.npz` matrices derived from individual-level data.
