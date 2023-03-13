import logging

import click
import networkx as nx

from cleora.cleora import Cleora

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_networkx_graph_from_file(filepath) -> nx.Graph:
    """Load a networkx graph from a file"""
    logger.info("Loading graph from file")
    graph = nx.read_edgelist(filepath, nodetype=int)
    logger.info("Graph loaded")
    return graph


@click.command()
@click.option(
    "--filepath",
    default="data/example_1.txt",
    help="Path to the file containing the graph",
)
def main(filepath):
    graph = load_networkx_graph_from_file(filepath)

    cleora = Cleora()
    embedding = cleora.embed(graph)

    logger.info("Embedding: %s", embedding)


if __name__ == "__main__":
    main()
