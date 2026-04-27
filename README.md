# Discrete PSO for Order Batching: A Comparative Study of Binary and Set-Based Solution Representations
This repository contains the source code for the algorithms used in the project, as well as the datasets used (provided by [Mercado Livre](https://github.com/mercadolibre/challenge-sbpo-2025/tree/master)).

The algorithms used were:
- Set-Based PSO (SBPSO) [[Langeveld and Engelbrecht, 2012]](https://www.researchgate.net/publication/257722514_Set-based_particle_swarm_optimization_applied_to_the_multidimensional_knapsack_problem)
- Modified Binary PSO (MBPSO) [[Yang et al., 2014]](https://ieeexplore.ieee.org/document/6661359)
- Modified Binary PSO with Zhang's transfer function (MBPSOzt) [[Zhang et al., 2020]](https://link.springer.com/chapter/10.1007/978-3-030-53956-6_18)

MBPSOzt was an adaptation, in which the transfer function of the original algorithm was replaced with one capable of controlling its opening and thus modifying the velocity-to-probability mapping.

## Experiments Design
The experiments were developed in Python 3.14.4 and run on an AMD Ryzen 5 7520U processor with 8 GB of memory, using Arch Linux. Runs were performed sequentially, rebooted between variants to reduce hardware interference and ensure consistent timing.

The 2025 SBPO challenge dataset consists of 20 instances. These instances vary in orders ($o$), item variety ($i$), ans aisles ($a$), as shown in the table below. Each method was executed 30 times per instance using fixed pseudorandom seeds in the range [10,000, 99,999], generated via the "True Random Number Generator" available at [Random.org](https://www.random.org/), totaling 1,800 runs.

> [!note] Used seeds
> [10447, 22022, 24675, 35446, 35476, 37983, 39628, 40694, 41383, 42738, 45786, 46223, 48679, 56429, 56927, 58565, 60163, 61820, 61975, 65439, 68036, 68071, 72427, 73641, 78884, 82757, 85722, 87599, 94860, 95990]

| Dataset | Orders | Items | Aisles |
| ------- | ------ | ----- | ------ |
| 01 | 61 | 155 | 116 |
| 02 | 7 | 7 | 33 |
| 03 | 82 | 246 | 124 |
| 04 | 16 | 59 | 91 |
| 05 | 2625 | 6407 | 161 |
| 06 | 10341 | 7089 | 184 |
| 07 | 8320 | 5747 | 180 |
| 08 | 2185 | 5831 | 168 |
| 09 | 70 | 222 | 304 |
| 10 | 1602 | 3689 | 383 |
| 11 | 1029 | 2784 | 375 |
| 12 | 133 | 337 | 342 |
| 13 | 8375 | 7525 | 413 |
| 14 | 12402 | 10974 | 413 |
| 15 | 7367 | 6633 | 402 |
| 16 | 1108 | 1051 | 88 |
| 17 | 417 | 411 | 83 |
| 18 | 2682 | 2309 | 90 |
| 19 | 2257 | 2104 | 134 |
| 20 | 5 | 5 | 5 |

The parameters used are all hard-coded in the `main.py` file.

## Repository structure
At the root level, there is a file named `main.py`, which provides a simpler interface for running the selected method with all the parameters used in the article, based on the specified seed. The dataset must be located in the `datasets` directory, and the result will be saved in the `results` directory in the format required by the challenge.

In the `process` directory, there is a file named `dataset.py` that reads the specified dataset and loads the data it contains. All algorithms receive an instance of the class contained within it, as the results are saved directly to the `result` attribute.

Finally, the `methods` directory contains the code for the algorithms; the binary algorithms are in the `mbpso.py` file, and the set-based algorithm is in `sbpso.py`. The `utils.py` file contains code used in all the algorithms.