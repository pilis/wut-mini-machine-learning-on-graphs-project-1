from unittest import TestCase

import networkx as nx

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
