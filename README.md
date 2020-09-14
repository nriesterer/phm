Probability Heuristics Model
============================

The Probability Heuristics Model (PHM) is a model formalization of the probabilistic cognitive theory on human reasoning. The goal of this repository is to provide a general implementation of the formalism as described in the introductory articles:

> Chater, N., & Oaksford, M. (1999). The probability heuristics model of syllogistic reasoning. Cognitive psychology, 38(2), 191-258.
>
> Oaksford, M., & Chater, N. (2001). The probabilistic approach to human reasoning. Trends in cognitive sciences, 5(8), 349-357.

### Repository Content

- `benchmarks`: Sample CCOBRA benchmarks.
- `phm.py`: Implementation of the core PHM functionality.
- `phm_ccobra.py`: Model implementation for use in the [CCOBRA model evaluation framework](https://github.com/CognitiveComputationLab/ccobra).

### Requirements

- Python 3
- [numpy](https://numpy.org)
- [pandas](https://pandas.pydata.org)
- [ccobra](https://github.com/CognitiveComputationLab/ccobra) (only for `phm_ccobra.py`)

### Quickstart

To run the sample CCOBRA evaluations, download the repository and execute the following commands in a terminal:

```
cd /path/to/repository/benchmarks
ccobra adaption.json
ccobra coverage.json
ccobra prediction.json
```
