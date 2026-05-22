# Continuants

**Continuants: Toward a Theory of Behavioural Persistence in Long-Running LLM Agents**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20349035.svg)](https://doi.org/10.5281/zenodo.20349035)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Code and data accompanying a research paper that investigates whether long-running autonomous LLM agents accumulate behavioural state beyond what context replication captures.

## Paper

Published on Zenodo: [https://doi.org/10.5281/zenodo.20349035](https://doi.org/10.5281/zenodo.20349035)

OSF project (pre-registration and supplementary materials): [https://osf.io/vse2f/](https://osf.io/vse2f/)

## Citation

> Krishnan, P. (2026). *Continuants: Toward a Theory of Behavioural Persistence in Long-Running LLM Agents* (1.0). Zenodo. https://doi.org/10.5281/zenodo.20349035

See [CITATION.cff](./CITATION.cff) for the machine-readable version.

## What this repository contains

```
continuants/
├── src/                            # Agent infrastructure, probes, analysis pipeline
├── experiments/
│   └── drift_study/                # The two paired experiments reported in the paper
│       ├── run.py                  # Experiment runner
│       └── runs/                   # Raw outputs and analysis CSVs from completed runs
├── paper/                          # Paper source (markdown), bibliography, figures
├── CITATION.cff                    # Machine-readable citation
├── LICENSE                         # Apache 2.0 (code) — paper itself is CC BY 4.0 on Zenodo
└── README.md                       # You are here
```

## Reproducing the experiments

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your Anthropic API key in `.env` (see `.env.example`).
4. Run the drift study: `python -m experiments.drift_study.run --pairs 3 --tickets-per-session 20`
5. Analyse a completed run: `python -m src.eval.analyze --latest`
6. Score with LLM judges: `python -m src.eval.score_run_v2 --latest`

All experiments use fixed random seeds. Specific package versions are pinned in `requirements.txt`.

Per-arm budget cap is $20 USD by default. The full two-experiment design (E1 templated, E2 Bitext) takes roughly $60 in API costs and several hours of wall-clock time.

## Contributors

- **Prakash Krishnan**, Independent Researcher. ORCID: [0009-0007-5481-6178](https://orcid.org/0009-0007-5481-6178). Conceptualization, methodology, software, formal analysis, investigation, data curation, writing, visualization, project administration.
- **Tarun Karthikeyan**, Research Assistant. ORCID: [0009-0008-3611-9181](https://orcid.org/0009-0008-3611-9181). Methodology and investigation discussions on experimental design, the pairwise judge approach, and the New Guy Syndrome framing.

See [CONTRIBUTORS.md](./CONTRIBUTORS.md) for the full CRediT taxonomy breakdown.

## License

- **Code**: Apache 2.0 (see [LICENSE](./LICENSE))
- **Paper** (on Zenodo): Creative Commons Attribution 4.0 International (CC BY 4.0)

## Status

Version 1.0 published on Zenodo, May 22, 2026. The framework, experiments, and findings are stable. Follow-up papers in this research program are tracked in [RESEARCH_ROADMAP.md](./RESEARCH_ROADMAP.md).
