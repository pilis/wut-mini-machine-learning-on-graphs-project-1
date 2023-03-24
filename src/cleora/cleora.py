import logging

import networkx as nx
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Cleora:
    def __init__(self, num_iterations: int = 5, num_dimensions: int = 3):
        """Initialize the Cleora"""
        self.num_iterations = num_iterations
        self.num_dimensions = num_dimensions

    def embed(self, graph: nx.Graph) -> np.ndarray:
        """Embed a graph into a vector space"""
        transition_matrix = self._get_transition_matrix(graph)
        logging.debug(transition_matrix)
        # Initialize the embedding matrix
        num_nodes = len(graph.nodes())
        embedding_matrix = self._initialize_embedding_matrix(
            num_nodes, self.num_dimensions
        )

        logging.debug("Embedding matrix before training: ")
        logging.debug(embedding_matrix)

        # Train the embedding matrix
        for _ in range(self.num_iterations):
            embedding_matrix = self._train_embedding(
                transition_matrix, embedding_matrix
            )

        return embedding_matrix

    def _get_transition_matrix(self, graph: nx.Graph) -> np.ndarray:
        """Get the transition matrix for a graph"""
        # Get the adjacency matrix. The adjacency matrix is a square matrix that represents the graph's edges as 1s and 0s.
        adjacency_matrix = nx.to_numpy_array(graph)
        # Get the degree matrix. Degree matrix is a diagonal matrix that contains the degree of each node.
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        # Get the transition matrix. Transition matrix is the inverse of the degree matrix multiplied by the adjacency matrix.
        transition_matrix = np.linalg.inv(degree_matrix) @ adjacency_matrix
        return transition_matrix

    def _initialize_embedding_matrix(
        self, num_nodes: int, num_dimensions
    ) -> np.ndarray:
        """Initialize the embedding matrix with -1 and 1 using uniform distribution"""
        return np.random.uniform(-1, 1, size=(num_nodes, num_dimensions))

    def _train_embedding(
        self, transition_matrix: np.ndarray, embedding_matrix: np.ndarray
    ) -> np.ndarray:
        """Train the embedding matrix"""
        num_dimensions = embedding_matrix.shape[1]

        # Iterate over the columns of the embedding matrix
        for i in range(num_dimensions):
            # Multiply the transition matrix by the ith column of the embedding matrix
            embedding_matrix[:, i] = transition_matrix @ embedding_matrix[:, i]

        logging.debug("Embedding matrix before normalization: ")
        logging.debug(embedding_matrix)

        # Normalize the embedding matrix with L2 norm
        embedding_matrix_l2_norm = np.linalg.norm(
            embedding_matrix, axis=1, keepdims=True
        )
        logging.debug("Embedding matrix L2 norm: ")
        logging.debug(embedding_matrix_l2_norm)

        normalized_embedding_matrix = np.divide(
            embedding_matrix, embedding_matrix_l2_norm
        )

        return normalized_embedding_matrix
