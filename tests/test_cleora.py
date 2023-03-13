from unittest import TestCase

import networkx as nx
import numpy as np

from cleora.cleora import Cleora


class TestCleoraChunkNodes(TestCase):
    def test_should_return_all_nodes_in_single_chunk_when_num_chunks_is_one(self):
        graph = nx.Graph()
        graph.add_nodes_from([1, 2])
        graph.add_edges_from([(1, 2)])

        cleora = Cleora()
        chunks = cleora._chunk_graph(graph, num_chunks=1)
        chunks = list(chunks)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0].nodes()), 2)
        self.assertEqual(len(chunks[0].edges()), 1)

    def test_should_return_two_chunks_with_all_edges_when_num_chunks_is_two_and_graph_is_triangle(
        self,
    ):
        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3])
        graph.add_edges_from([(1, 2), (2, 3), (1, 3)])

        cleora = Cleora()
        chunks = cleora._chunk_graph(graph, num_chunks=2)
        chunks = list(chunks)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0].nodes()), 3)
        self.assertEqual(len(chunks[0].edges()), 3)
        self.assertEqual(len(chunks[1].nodes()), 3)
        self.assertEqual(len(chunks[1].edges()), 3)

    def test_should_return_two_chunks_with_different_edges_when_num_chunks_is_two_and_graph_is_line_on_three_nodes(
        self,
    ):
        graph = nx.Graph()
        graph.add_nodes_from([1, 2, 3])
        graph.add_edges_from([(1, 2), (2, 3)])

        cleora = Cleora()
        chunks = cleora._chunk_graph(graph, num_chunks=2)
        chunks = list(chunks)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0].nodes()), 3)
        self.assertEqual(len(chunks[0].edges()), 2)
        self.assertEqual(len(chunks[1].nodes()), 2)
        self.assertEqual(len(chunks[1].edges()), 1)


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
