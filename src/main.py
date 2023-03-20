import logging

import click
import networkx as nx
import numpy as np

from cleora.cleora import Cleora

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_networkx_graph_from_file(filepath: str) -> nx.Graph:
    """Load a networkx graph from a file"""
    logger.info("Loading graph from file")
    graph = nx.read_edgelist(filepath, nodetype=int)
    logger.info("Graph loaded")
    return graph


def save_embedding_to_file(embedding: np.ndarray, filepath: str) -> None:
    """Save the embedding to a file"""
    np.savetxt(filepath, embedding, delimiter=" ")


@click.command()
@click.option(
    "--input-filepath",
    default="data/facebook_combined.txt",
    help="Path to the input file containing the graph",
)
@click.option(
    "--output-filepath",
    default="data/facebook_combined_emb.txt",
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
    input_filepath: str, output_filepath: str, num_dimensions: int, num_iterations: int
) -> None:
    """Main function to run the Cleora algorithm"""
    graph = load_networkx_graph_from_file(input_filepath)

    cleora = Cleora(num_dimensions=num_dimensions, num_iterations=num_iterations)
    embedding = cleora.embed(graph)

    save_embedding_to_file(embedding, output_filepath)


if __name__ == "__main__":
    main()
