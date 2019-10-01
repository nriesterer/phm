Probability Heuristics Model
============================

The Probability Heuristics Model (PHM) is a model formalization of the probabilistic cognitive theory on human reasoning. The goal of this repository is to provide a general implementation of the formalism as described in the introductory articles:

> Chater, N., & Oaksford, M. (1999). The probability heuristics model of syllogistic reasoning. Cognitive psychology, 38(2), 191-258.
>
> Oaksford, M., & Chater, N. (2001). The probabilistic approach to human reasoning. Trends in cognitive sciences, 5(8), 349-357.

### Repository Content

- `phm.py`: Implementation of the core PHM functionality
- `phm_ccobra.py`: Model implementation for use in the [CCOBRA model evaluation framework](https://github.com/CognitiveComputationLab/ccobra).

### Requirements

- Python 3
- [numpy](https://numpy.org)
- [pandas](https://pandas.pydata.org)
- [ccobra](https://github.com/CognitiveComputationLab/ccobra) (only for `phm_ccobra.py`)

### Testing the model in CCOBRA

To test the model in the CCOBRA model evaluation framework, simply navigate to the online evaluation website [https://orca.informatik.uni-freiburg.de/ccobra/index.php?sel=syl](https://orca.informatik.uni-freiburg.de/ccobra/index.php?sel=syl) and upload the repository content as a `.zip`-file (the evaluation takes around 6 minutes).
