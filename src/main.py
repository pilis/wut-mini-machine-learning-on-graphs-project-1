import logging
import os

import click
import networkx as nx
import numpy as np

from cleora.cleora import (
    Cleora,
    CleoraFixedIterations,
    CleoraNeighbourhoodDepthIterations,
)

is_debug_mode = os.getenv("DEBUG", "false") == "true"
logging_level = logging.DEBUG if is_debug_mode else logging.INFO
logging.basicConfig(level=logging_level, force=is_debug_mode)
if is_debug_mode:
    logging.debug("Debug mode is on")


def load_networkx_graph_from_file(filepath: str) -> nx.Graph:
    """Load a networkx graph from a file"""
    logging.debug("Loading graph from file")
    graph = nx.read_edgelist(filepath, nodetype=int)
    logging.debug("Graph loaded")
    return graph


def save_embedding_to_file(embedding: np.ndarray, filepath: str) -> None:
    """Save the embedding to a file"""
    np.savetxt(filepath, embedding, delimiter=" ")


@click.command()
@click.option(
    "--algorithm-version",
    type=click.Choice(["fixed", "neighbourhood_depth"], case_sensitive=False),
    default="fixed",
    help="Choose the algorithm version to run",
)
@click.option(
    "--input-filepath",
    default="data/graphs/triangle.txt",
    help="Path to the input file containing the graph",
)
@click.option(
    "--output-filepath",
    default="data/embeddings/triangle.txt",
    help="Path to the output file containing the embedding",
)
@click.option(
    "--num_dimensions",
    default=5,
    help="Number of dimensions for the embedding",
)
@click.option(
    "--num_iterations",
    default=3,
    help="Number of iterations to run the algorithm",
)
def main(
    algorithm_version: str,
    input_filepath: str,
    output_filepath: str,
    num_dimensions: int,
    num_iterations: int,
) -> None:
    """Main function to run the Cleora algorithm"""

    def get_cleora_instance(algorithm_version: str) -> Cleora:
        if algorithm_version == "fixed":
            return CleoraFixedIterations(
                num_dimensions=num_dimensions, num_iterations=num_iterations
            )
        elif algorithm_version == "neighbourhood_depth":
            return CleoraNeighbourhoodDepthIterations(num_dimensions=num_dimensions)
        else:
            raise ValueError(f"Invalid algorithm_version version: {algorithm_version}")

    graph = load_networkx_graph_from_file(input_filepath)

    cleora = get_cleora_instance(algorithm_version)
    embedding = cleora.embed(graph)

    save_embedding_to_file(embedding, output_filepath)


if __name__ == "__main__":
    main()
