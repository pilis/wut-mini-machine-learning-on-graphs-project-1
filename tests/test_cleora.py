from unittest import TestCase

import networkx as nx
import numpy as np

from cleora.cleora import Cleora


class TestCleoraGetTransitionMatrix(TestCase):
    def test_should_return_transition_matrix_for_triangle(self):
        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3])
        graph.add_edges_from([(1, 2), (2, 3), (1, 3)])

        cleora = Cleora()
        transition_matrix = cleora._get_transition_matrix(graph)

        expected_transition_matrix = [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
        self.assertEqual(transition_matrix.tolist(), expected_transition_matrix)


class TestCleoraInitializeEmbeddingMatrix(TestCase):
    def test_should_return_matrix_with_correct_shape_and_cell_values_to_be_minus_one_or_one(
        self,
    ):
        num_nodes = 3
        num_dimensions = 2

        cleora = Cleora()
        embedding_matrix = cleora._initialize_embedding_matrix(
            num_nodes, num_dimensions
        )

        self.assertEqual(embedding_matrix.shape, (num_nodes, num_dimensions))
        self.assertTrue(
            (embedding_matrix == -1).any() and (embedding_matrix == 1).any()
        )


class TestCleoraTrainEmbedding(TestCase):
    def test_should_return_embedding_matrix_with_correct_shape(self):
        transition_matrix = np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        )
        embedding_matrix = np.array(
            [
                [1, -1],
                [-1, 1],
                [1, -1],
            ]
        )

        cleora = Cleora()
        embedding_matrix = cleora._train_embedding(transition_matrix, embedding_matrix)
        self.assertEqual(embedding_matrix.shape, (3, 2))
