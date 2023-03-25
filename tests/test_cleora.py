from unittest import TestCase

import networkx as nx
import numpy as np

from cleora.cleora import (
    Cleora,
    CleoraFixedIterations,
    CleoraNeighbourhoodDepthIterations,
)


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
        self.assertTrue(np.all(embedding_matrix >= -1))
        self.assertTrue(np.all(embedding_matrix <= 1))


class TestCleoraFixedIterationsEmbed(TestCase):
    def test_should_return_embedding_matrix_with_correct_shape(self):
        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3])
        graph.add_edges_from([(1, 2), (2, 3), (1, 3)])

        num_iterations = 5
        num_dimensions = 2

        cleora = CleoraFixedIterations(num_iterations, num_dimensions)
        embedding_matrix = cleora.embed(graph)

        self.assertEqual(embedding_matrix.shape, (len(graph.nodes()), num_dimensions))


class TestCleoraNeighbourhoodDepthIterationsGetNodesNeighbourhoodDepth(TestCase):
    def test_should_return_neighbourhood_depth_matrix_for_triangle(self):
        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3])
        graph.add_edges_from([(1, 2), (2, 3), (1, 3)])

        cleora = CleoraNeighbourhoodDepthIterations()
        nodes_neighbourhood_depth = cleora._get_nodes_neighbourhood_depth(graph)

        expected_nodes_neighbourhood_depth = [1.0, 1.0, 1.0]
        self.assertListEqual(
            list(nodes_neighbourhood_depth), expected_nodes_neighbourhood_depth
        )

    def test_should_return_nodes_neighbourhood_depth_for_path(self):
        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3])
        graph.add_edges_from([(1, 2), (2, 3)])

        cleora = CleoraNeighbourhoodDepthIterations()
        nodes_neighbourhood_depth = cleora._get_nodes_neighbourhood_depth(graph)

        expected_nodes_neighbourhood_depth = [2.0, 1.0, 2.0]
        self.assertListEqual(
            list(nodes_neighbourhood_depth), expected_nodes_neighbourhood_depth
        )

    def test_should_return_nodes_neighbourhood_depth_for_disconnected_graph(self):
        # A disconnected graph with 2 components and 3 nodes in first component and 2 nodes in second component
        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3, 4, 5])
        graph.add_edges_from([(1, 2), (2, 3), (4, 5)])

        cleora = CleoraNeighbourhoodDepthIterations()
        nodes_neighbourhood_depth = cleora._get_nodes_neighbourhood_depth(graph)

        expected_nodes_neighbourhood_depth = [2.0, 1.0, 2.0, 1.0, 1.0]
        self.assertListEqual(
            list(nodes_neighbourhood_depth), expected_nodes_neighbourhood_depth
        )
