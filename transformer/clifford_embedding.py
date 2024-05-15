import torch
import torch


class NBodyGraphEmbedder:
    def __init__(self, clifford_algebra, make_edge_attr):
        self.clifford_algebra = clifford_algebra
        self.make_edge_attr = make_edge_attr

    def embed_nbody_graphs(self, batch):
        batch_size, n_nodes, _ = batch[0].size()
        full_node_embedding, full_edge_embedding, loc_end_clifford, edges = self.get_embedding(batch, batch_size,
                                                                                               n_nodes)
        attention_mask = self.get_attention_mask(batch_size, n_nodes, edges)

        return full_node_embedding, full_edge_embedding, loc_end_clifford, attention_mask

    def get_embedding(self, batch, batch_size, n_nodes):
        loc_mean, vel, edge_attr, charges, loc_end, edges = self.preprocess(batch)
        invariants = self.clifford_algebra.embed(charges, (0,))
        xv = torch.stack([loc_mean, vel], dim=1)
        covariants = self.clifford_algebra.embed(xv, (1, 2, 3))

        full_node_embedding = self.get_full_node_embedding(invariants, covariants, batch_size)
        start_nodes, end_nodes = self.get_edge_nodes(edges, n_nodes, batch_size)

        full_edge_embedding = self.get_full_edge_embedding(edge_attr, full_node_embedding, (start_nodes, end_nodes),
                                                           batch_size)
        loc_end_clifford = self.clifford_algebra.embed(loc_end, (1, 2, 3))

        return full_node_embedding, full_edge_embedding, loc_end_clifford, (start_nodes, end_nodes)

    def preprocess(self, batch):
        loc, vel, edge_attr, charges, loc_end, edges = batch
        loc_mean = self.compute_mean_centered(loc)
        loc_end_mean = self.compute_mean_centered(loc_end)  # check this!
        loc_mean, vel, edge_attr, charges, loc_end_mean = self.flatten_tensors(loc_mean, vel, edge_attr, charges,
                                                                               loc_end_mean)
        return loc_mean, vel, edge_attr, charges, loc_end_mean, edges

    def compute_mean_centered(self, tensor):
        return tensor - tensor.mean(dim=1, keepdim=True)

    def flatten_tensors(self, *tensors):
        return [tensor.float().view(-1, *tensor.shape[2:]) for tensor in tensors]

    def get_full_node_embedding(self, invariants, covariants, batch_size):
        nodes_in_clifford = torch.cat([invariants[:, None], covariants], dim=1)
        zero_padding_nodes = torch.zeros(5 * batch_size, 4, 8)
        return torch.cat((nodes_in_clifford, zero_padding_nodes), dim=1)

    def get_edge_nodes(self, edges, n_nodes, batch_size):
        batch_index = torch.arange(batch_size, device=edges.device)
        edges = edges + n_nodes * batch_index[:, None, None]
        edges = tuple(edges.transpose(0, 1).flatten(1))
        return edges

    def get_full_edge_embedding(self, edge_attr, nodes_in_clifford, edges, batch_size):
        orig_edge_attr_clifford = self.clifford_algebra.embed(edge_attr[..., None], (0,))
        extra_edge_attr_clifford = self.make_edge_attr(nodes_in_clifford, edges)
        edge_attr_all = torch.cat((orig_edge_attr_clifford, extra_edge_attr_clifford), dim=1)
        zero_padding_edges = torch.zeros(20 * batch_size, 3, 8)
        return torch.cat((zero_padding_edges, edge_attr_all), dim=1)

    def make_edge_attr(self, node_features, edges):
        node1_features = node_features[edges[0]]
        node2_features = node_features[edges[1]]
        edge_attributes = node1_features - node2_features
        return edge_attributes

    def get_attention_mask(self, batch_size, n_nodes, edges):
        num_edges_per_graph = edges.size(1) // batch_size
        total_elements_per_graph = n_nodes + num_edges_per_graph

        attention_mask = torch.full((batch_size * total_elements_per_graph, batch_size * total_elements_per_graph),
                                    float('-inf'), device=edges.device)

        batch_index = torch.arange(batch_size, device=edges.device).view(-1, 1, 1)
        node_start_indices = batch_index * total_elements_per_graph
        edge_start_indices = node_start_indices + n_nodes

        start_nodes = edges[0] + node_start_indices
        end_nodes = edges[1] + node_start_indices
        edge_indices = torch.arange(num_edges_per_graph, device=edges.device).view(1, -1) + edge_start_indices

        start_nodes = start_nodes.flatten()
        end_nodes = end_nodes.flatten()
        edge_indices = edge_indices.flatten()

        attention_mask[edge_indices[:, None], start_nodes] = 0
        attention_mask[edge_indices[:, None], end_nodes] = 0
        attention_mask[start_nodes[:, None], edge_indices] = 0
        attention_mask[end_nodes[:, None], edge_indices] = 0

        for b in range(batch_size):
            node_start_idx = b * total_elements_per_graph
            node_end_idx = node_start_idx + n_nodes
            attention_mask[node_start_idx:node_end_idx, node_start_idx:node_end_idx] = 0

        attention_mask = attention_mask.float()
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))

        return attention_mask




