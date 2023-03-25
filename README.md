# WUT MiNI Machine Learning on Graphs 2022/2023 Summer Project

## Project description

Goal of this project is to implement a selected node embedding algorithm for graphs and then propose a modification of the algorithm to improve its performance.

[Cleora](https://github.com/Synerise/cleora) is a general-purpose model for efficient, scalable learning of stable and inductive entity embeddings for heterogeneous relational data. Cleora embeds entities in n-dimensional spherical spaces utilizing extremely fast stable, iterative random projections, which allows for unparalleled performance and scalability.

### Improvement proposal

Background: The iteration number defines the breadth of neighborhood on which a single node is averaged: iteration number i means that nodes with similar i-hop neighborhoods will have similar representations. The iteration number is related to the concept of average path length from the area of graph topology. The average path length is defined as the average number of steps along the shortest paths for all possible pairs of network nodes. If the iteration number reaches the average path length, an average node will likely have access to all other nodes.  Thus, iteration number slightly exceeding the average path length can be deemed optimal.

Hypothesis: If we base the number of iterations necessary for a given node on the depth of its neighborhood instead of using the average path length for all nodes, we can identify the optimal number of iterations for that node and therefore avoid overfitting behaviour as shown in the paper when too high iteration number was picked.

## Getting started

### Prerequisites

- Python 3.11.0+
- pre-commit 3.1.1

### Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements-dev.txt -r src/requirements.txt
```

Install pre-commit hooks:

```bash
pre-commit install
```

## Usage

### Running the code

Run the script with default parameters:
```bash
python src/main.py
```

To run the script in debug mode set the `DEBUG` environment variable to `True`:
```bash
DEBUG=True python src/main.py
```

Optional parameters:
- `--algorithm-version` - version of the algorithm to run. Possible values: `fixed`, `neighbourhood_depth`. Default: `fixed`.
- `--input-filepath` - path to the input file. Default: `data/graphs/triangle.txt`.
- `--output-filepath` - path to the output file. Default: `data/embeddings/triangle.txt`.
- `num_dimensions` - number of dimensions of the embedding. Default: 5
- `num_iterations` - number of iterations of the algorithm. Default: 3

## Contributing

Running tests:

```bash
pytest
```

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Authors

- Piotr Pilis
- Anish Gupta
