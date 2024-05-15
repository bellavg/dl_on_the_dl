import torch
from original_models.modules.linear import MVLinear

from algebra.cliffordalgebra import CliffordAlgebra

class NBodyGraphEmbedder:
    def __init__(self, clifford_algebra, in_features, embed_dim):
        self.clifford_algebra = clifford_algebra
        self.embedding = MVLinear(
            self.clifford_algebra, in_features,  embed_dim, subspaces=False
        )

    def embed_nbody_graphs(self, batch):
        batch_size, n_nodes, _ = batch[0].size()
        full_node_embedding, full_edge_embedding, loc_end_clifford, edges = self.get_embedding(batch, batch_size, n_nodes)
        attention_mask = self.get_attention_mask(batch_size, n_nodes, edges)
        return full_node_embedding, full_edge_embedding, loc_end_clifford, attention_mask

    def get_embedding(self, batch, batch_size, n_nodes):
        loc_mean, vel, edge_attr, charges, loc_end, edges = self.preprocess(batch)


        # Embed data in Clifford space
        invariants = self.clifford_algebra.embed(charges, (0,))
        xv = torch.stack([loc_mean, vel], dim=1)
        covariants = self.clifford_algebra.embed(xv, (1, 2, 3))

        full_node_embedding = torch.cat([invariants[:, None], covariants], dim=1)

        # Get edge nodes and edge features
        start_nodes, end_nodes = self.get_edge_nodes(edges, n_nodes, batch_size)
        full_edge_embedding = self.get_full_edge_embedding(edge_attr, full_node_embedding, (start_nodes, end_nodes), batch_size, n_nodes)

        # Clifford embeddings for end locations
        loc_end_clifford = self.clifford_algebra.embed(loc_end, (1, 2, 3))

        return full_node_embedding, full_edge_embedding, loc_end_clifford, (start_nodes, end_nodes)

    def preprocess(self, batch):
        loc, vel, edge_attr, charges, loc_end, edges = batch
        #print("before",loc.shape, vel.shape, edge_attr.shape, edges.shape, charges.shape)
        loc_mean = self.compute_mean_centered(loc)
        loc_end_mean = self.compute_mean_centered(loc_end)
        loc_mean, vel, edge_attr, charges, loc_end_mean = self.flatten_tensors(loc_mean, vel, edge_attr, charges, loc_end_mean)
        return loc_mean, vel, edge_attr, charges, loc_end_mean, edges

    def compute_mean_centered(self, tensor):
        return tensor - tensor.mean(dim=1, keepdim=True)

    def flatten_tensors(self, *tensors):
        return [tensor.float().view(-1, *tensor.shape[2:]) for tensor in tensors]


    def get_edge_nodes(self, edges, n_nodes, batch_size):
        batch_index = torch.arange(batch_size, device=edges.device)
        edges = edges + n_nodes * batch_index[:, None, None]
        edges = tuple(edges.transpose(0, 1).flatten(1))
        return edges

    def get_full_edge_embedding(self, edge_attr, nodes_in_clifford, edges, batch_size, n_nodes):
        orig_edge_attr_clifford = self.clifford_algebra.embed(edge_attr[..., None], (0,))
        extra_edge_attr_clifford = self.make_edge_attr(nodes_in_clifford, edges)
        edge_attr_all = torch.cat((orig_edge_attr_clifford, extra_edge_attr_clifford), dim=1)

        # Dynamically calculate the number of edges
        num_edges = batch_size * n_nodes * (n_nodes - 1) // 2
        zero_padding_edges = torch.zeros(num_edges, 3, 8, device=edge_attr_all.device)
        edge_attr_all = torch.cat((zero_padding_edges, edge_attr_all), dim=1)

        # Project the edge features to higher dimensions
        projected_edges = self.edge_projection(edge_attr_all)
        return projected_edges

    def make_edge_attr(self, node_features, edges):
        node1_features = node_features[edges[0]]
        node2_features = node_features[edges[1]]
        edge_attributes = node1_features - node2_features
        return edge_attributes

    def get_attention_mask(self, batch_size, n_nodes, edges):
        # Calculate the number of edges per graph
        num_edges_per_graph = n_nodes * (n_nodes - 1) // 2

        # Initialize an attention mask with zeros
        total_elements_per_graph = n_nodes + num_edges_per_graph
        attention_mask = torch.zeros(batch_size, total_elements_per_graph, total_elements_per_graph,
                                     device=edges[0].device)

        for b in range(batch_size):
            node_start_idx = b * total_elements_per_graph
            edge_start_idx = node_start_idx + n_nodes

            for i in range(num_edges_per_graph):
                start_node = edges[0][i] + node_start_idx
                end_node = edges[1][i] + node_start_idx
                edge_idx = edge_start_idx + i

                # Edges can attend to their corresponding nodes
                attention_mask[b, edge_idx, start_node] = 1
                attention_mask[b, edge_idx, end_node] = 1

                # Nodes can attend to their corresponding edges
                attention_mask[b, start_node, edge_idx] = 1
                attention_mask[b, end_node, edge_idx] = 1

        # Convert the mask to float and set masked positions to -inf and allowed positions to 0
        attention_mask = attention_mask.float()
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))

        return attention_mask

