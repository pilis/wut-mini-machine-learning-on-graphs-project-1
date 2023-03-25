import logging
from collections import deque

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


class CleoraNeighbourhoodDepthIterations(Cleora):
    def __init__(self, num_dimensions: int = 3):
        """Initialize the Cleora"""
        super().__init__(num_dimensions)

    def embed(self, graph: nx.Graph) -> np.ndarray:
        """Embed a graph into a vector space"""
        num_nodes = len(graph.nodes())

        transition_matrix = self._get_transition_matrix(graph)
        embedding_matrix = self._initialize_embedding_matrix(
            num_nodes, self.num_dimensions
        )
        nodes_neighbourhood_depth = self._get_nodes_neighbourhood_depth(graph)

        # Iterate over the max neighbourhood depth in the graph
        max_nodes_neighbourhood_depth = np.max(nodes_neighbourhood_depth)
        logging.debug(f"Max nodes neighbourhood depth: {max_nodes_neighbourhood_depth}")
        for current_iteration in range(max_nodes_neighbourhood_depth):
            logging.debug(f"Current iteration: {current_iteration}")
            # Generate mask on transition matrix for nodes with neighbourhood depth less than or equal to i to stop updating their embeddings
            update_mask = nodes_neighbourhood_depth > current_iteration

            # Iterate over the columns of the embedding matrix
            for column_idx in range(self.num_dimensions):
                # Multiply the transition matrix by the ith column of the embedding matrix
                updated_dimension = transition_matrix @ embedding_matrix[:, column_idx]
                # Update the embedding matrix with the updated dimension only when update_mask is True
                embedding_matrix[:, column_idx] = np.where(
                    update_mask, updated_dimension, embedding_matrix[:, column_idx]
                )

            # Normalize the embedding matrix with L2 norm
            embedding_matrix_l2_norm = np.linalg.norm(
                embedding_matrix, axis=1, keepdims=True
            )
            updated_embedding = np.divide(embedding_matrix, embedding_matrix_l2_norm)

            # Update the embedding matrix with the updated embedding only when update_mask is True
            embedding_matrix[update_mask] = updated_embedding[update_mask]

        return embedding_matrix

    def _get_nodes_neighbourhood_depth(self, graph: nx.Graph) -> np.ndarray:
        """Get the neighbourhood depth of each node in the graph"""

        def get_max_bfs_depth(adjacency_matrix: np.ndarray, start_node: int) -> int:
            n = len(adjacency_matrix)
            visited = [False] * n
            depth = [0] * n
            queue = deque([start_node])

            visited[start_node] = True
            depth[start_node] = 0

            while queue:
                current_node = queue.popleft()

                for neighbor, is_adjacent in enumerate(adjacency_matrix[current_node]):
                    if is_adjacent and not visited[neighbor]:
                        visited[neighbor] = True
                        depth[neighbor] = depth[current_node] + 1
                        queue.append(neighbor)

            max_depth = max(depth)
            return max_depth

        adjacency_matrix = nx.to_numpy_array(graph)

        num_nodes = len(adjacency_matrix)
        nodes_neighbourhood_depth = np.zeros(num_nodes, dtype=int)
        logging.debug(f"Num nodes: {num_nodes}")

        for i in range(num_nodes):
            max_depth = get_max_bfs_depth(adjacency_matrix, i)
            nodes_neighbourhood_depth[i] = max_depth
            logging.debug(f"Current node: {i}, max depth: {max_depth}")

        logging.debug(f"Nodes neighbourhood depth: {nodes_neighbourhood_depth}")

        return nodes_neighbourhood_depth
