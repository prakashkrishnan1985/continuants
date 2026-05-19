# Continuants

**Continuants: Toward a Theory of Behavioural Persistence in Long-Running LLM Agents**

Code and data accompanying a research paper that investigates whether long-running autonomous LLM agents accumulate behavioural state beyond what context replication captures.

## What this repository contains

This repository holds the *code and data* for the experiments described in the paper. The paper itself is deposited on Zenodo (with DOI) and is not stored here.

```
continuants/
├── src/                            # Shared code and utilities
├── experiments/
│   └── <NN-name>/                  # One folder per experiment
│       ├── run.py                  # Experiment runner
│       └── data/                   # Raw outputs (after running)
├── CITATION.cff                    # How to cite this work
├── LICENSE                         # Apache 2.0
└── README.md                       # You are here
```

## Paper

The working paper is deposited on Zenodo: *DOI to be added on publication.*

OSF project (pre-registration and supplementary materials): https://osf.io/vse2f/

## Author

Prakash Krishnan, Independent Researcher.
ORCID: [0009-0007-5481-6178](https://orcid.org/0009-0007-5481-6178)

## Reproducing the experiments

*Will be filled in once experiment code is committed.*

The high-level reproduction flow will be:

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run an individual experiment: `python -m experiments.<NN-name>.run`
4. Analysis scripts and plots will be in `experiments/<NN-name>/analysis/`.

All experiments use fixed random seeds. Specific package versions are pinned in `requirements.txt` (added once code is committed).

## How to cite

See [CITATION.cff](./CITATION.cff). The Zenodo deposit provides a canonical DOI for the paper; that is the preferred citation target once it is live.

## License

Apache 2.0 (see [LICENSE](./LICENSE)). The paper itself, deposited on Zenodo, is released under CC BY 4.0.

## Status

Work-in-progress. First paper deposit expected Saturday May 24, 2026. This README and the repository structure will evolve as experiments are designed, run, and analyzed.
