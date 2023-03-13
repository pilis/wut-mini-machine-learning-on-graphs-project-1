import math
from typing import Generator

import networkx as nx
import numpy as np


class Cleora:
    def __init__(self):
        """Initialize the Cleora class with config"""
        pass

    def embed(self, graph: nx.Graph) -> np.ndarray:
        """Embed a graph into a vector space"""
        chunks = self._chunk_graph(graph)

        for chunk in chunks:
            transition_matrix = self._get_transition_matrix(chunk)  # noqa: F841
            pass

    def _chunk_graph(
        self, graph: nx.Graph, num_chunks: int = 1
    ) -> Generator[nx.Graph, None, None]:
        """Chunk a graph into subgraphs"""
        if not num_chunks > 0:
            raise ValueError("num_chunks must be greater than 0")
        # Determine the number of nodes in the graph
        num_nodes = len(graph.nodes())
        # Determine the size of each chunk
        # If the number of nodes is not evenly divisible by the number of chunks, the last chunk will be smaller
        chunk_size = math.ceil(num_nodes / num_chunks)

        # Generate a subgraphs
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            if i == num_chunks - 1:
                end = num_nodes
            nodes = list(graph.nodes())[start:end]
            # Add all neighbors of the nodes in the chunk to the chunk
            neighbors = []
            for node in nodes:
                neighbors.extend(list(graph.neighbors(node)))
            nodes.extend(neighbors)
            # Get induced subgraph
            subgraph = graph.subgraph(nodes)
            yield subgraph

    def _get_transition_matrix(self, graph: nx.Graph) -> np.ndarray:
        """Get the transition matrix for a graph"""
        # Get the adjacency matrix
        adjacency_matrix = nx.to_numpy_array(graph)
        # Get the degree matrix
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        # Get the transition matrix
        transition_matrix = np.linalg.inv(degree_matrix) @ adjacency_matrix
        return transition_matrix
