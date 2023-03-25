import networkx as nx
import numpy as np


class Cleora:
    def __init__(self, num_dimensions: int = 3):
        """Initialize the Cleora"""
        self.num_dimensions = num_dimensions

    def embed(self, graph: nx.Graph) -> np.ndarray:
        """Embed a graph into a vector space"""
        raise NotImplementedError

    def _initialize_embedding_matrix(
        self, num_nodes: int, num_dimensions: int
    ) -> np.ndarray:
        """Initialize the embedding matrix with -1 and 1 using uniform distribution"""
        return np.random.uniform(-1, 1, size=(num_nodes, num_dimensions))

    def _get_transition_matrix(self, graph: nx.Graph) -> np.ndarray:
        """Get the transition matrix for a graph"""
        adjacency_matrix = nx.to_numpy_array(graph)
        degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
        transition_matrix = np.linalg.inv(degree_matrix) @ adjacency_matrix
        return transition_matrix


class CleoraFixedIterations(Cleora):
    def __init__(self, num_iterations: int = 5, num_dimensions: int = 3):
        """Initialize the Cleora"""
        super().__init__(num_dimensions)

        self.num_iterations = num_iterations

    def embed(self, graph: nx.Graph) -> np.ndarray:
        """Embed a graph into a vector space"""
        num_nodes = len(graph.nodes())

        transition_matrix = self._get_transition_matrix(graph)
        embedding_matrix = self._initialize_embedding_matrix(
            num_nodes, self.num_dimensions
        )

        # Iterate over the number of iterations
        for _ in range(self.num_iterations):
            # Iterate over the columns of the embedding matrix
            for i in range(self.num_dimensions):
                # Multiply the transition matrix by the ith column of the embedding matrix
                embedding_matrix[:, i] = transition_matrix @ embedding_matrix[:, i]

            # Normalize the embedding matrix with L2 norm
            embedding_matrix_l2_norm = np.linalg.norm(
                embedding_matrix, axis=1, keepdims=True
            )
            embedding_matrix = np.divide(embedding_matrix, embedding_matrix_l2_norm)

        return embedding_matrix
