# track_scores

A Python implementation for the objective evaluation and comparison of automated thunderstorm identification and tracking algorithms, based on the scoring methodology introduced by [Satrio and Calhoun (2022)](https://journals.ametsoc.org/view/journals/wefo/37/11/WAF-D-22-0047.1.xml). This package computes a composite skill score across four performance parameters and includes a best track reanalysis module based on the methods described by [Lakshmanan and Smith (2010)](https://journals.ametsoc.org/view/journals/wefo/25/2/2009waf2222330_1.xml).

## Overview

Storm identification and tracking algorithms are widely used in both operational forecasting and research, yet no single standard method exists for objectively determining their performance. This repository implements a comparative skill score framework that quantifies algorithm performance using four parameters:

1. **Size Consistency**: Measures the temporal consistency of identified storm object areas along each track, rewarding algorithms that produce stable object sizes over time.
2. **Track Linearity**: Evaluates how linear (physically realistic) the resulting storm tracks are, penalising erratic jumps or non-physical trajectory changes.
3. **Mean Track Duration**: Quantifies the average lifespan of tracked storms, favouring algorithms that maintain longer, more meaningful tracks and minimise short-lived false detections.
4. **Best Track Score**: Compares the algorithm's output against an optimal post-event reanalysis (best track), scoring how well the original tracking matches a corrected reference.

## Best Track Reanalysis

The `best_track.py` module implements the best track reanalysis methodology from [Lakshmanan and Smith (2010)](https://journals.ametsoc.org/view/journals/wefo/25/2/2009waf2222330_1.xml). Best track acts as a post-event optimisation of storm tracks; it fixes track breaks and prunes false detections of short-lived tracks from the original algorithm output. The best track reanalysis is then compared against the original algorithm output using a point-based scoring system:

| Condition | Points |
|---|---|
| Object exists in best track and is correctly associated with the best track trajectory | +1.0 |
| Object exists in best track but is reassociated to a different best track trajectory | +0.5 |
| Object does not exist in the best track reanalysis (i.e., it is dropped) | +0.0 |

The total points are divided by the number of original storm objects, such that a perfect algorithm (where the best track reanalysis matches the original) receives a score of 1.0.

## Repository Structure

```
track_scores/
├── best_track.py          # Best track reanalysis implementation
├── readers.py             # Data readers for algorithm output formats
├── track_scores.ipynb     # Jupyter notebook demonstrating the scoring workflow
├── environment.yml        # Conda environment specification
└── dev/                   # Development and testing scripts
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/joshua-wx/track_scores.git
cd track_scores
```

2. Create the conda environment:

```bash
conda env create -f environment.yml
conda activate track_scores
```

## Usage

The primary workflow is demonstrated in the `track_scores.ipynb` Jupyter notebook. Open it to see a step-by-step example of loading algorithm output, computing the four scoring parameters, and generating the composite skill score.

```bash
jupyter notebook track_scores.ipynb
```

## References

- **Satrio, C. N., and K. M. Calhoun**, 2022: An Objective Scoring Method for Evaluating the Comparative Performance of Automated Storm Identification and Tracking Algorithms. *Weather and Forecasting*, **37**, 2107–2116, [https://doi.org/10.1175/WAF-D-22-0047.1](https://doi.org/10.1175/WAF-D-22-0047.1)

- **Lakshmanan, V., and T. Smith**, 2010: An Objective Method of Evaluating and Devising Storm-Tracking Algorithms. *Weather and Forecasting*, **25**, 701–709, [https://doi.org/10.1175/2009WAF2222330.1](https://doi.org/10.1175/2009WAF2222330.1)

## Contact

This project is maintained by [Joshua Soderholm](https://github.com/joshua-wx). For questions or issues, please use the [GitHub issue tracker](https://github.com/joshua-wx/track_scores/issues).