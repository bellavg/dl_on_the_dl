import torch
from original_models.modules.linear import MVLinear

from algebra.cliffordalgebra import CliffordAlgebra


class NBodyGraphEmbedder:
    def __init__(self, clifford_algebra, in_features, embed_dim):
        self.clifford_algebra = clifford_algebra
        self.node_projection = MVLinear(
            self.clifford_algebra, in_features, embed_dim, subspaces=False
        )
        self.edge_projection = MVLinear(
            self.clifford_algebra, 10, embed_dim, subspaces=False
        )

    def embed_nbody_graphs(self, batch):
        batch_size, n_nodes, _ = batch[0].size()
        loc_mean, vel, edge_attr, charges, loc_end, edges = self.preprocess(batch)
        full_node_embedding, nodes_stack = self.get_node_embedding(loc_mean, vel, charges)
        full_edge_embedding, edges = self.get_edge_embedding(edge_attr, edges, batch_size, n_nodes, nodes_stack)

        attention_mask = self.get_attention_mask(batch_size, n_nodes, edges)
        full_embedding = torch.cat((full_node_embedding, full_edge_embedding), dim=0)

        # Clifford embeddings for end locations
        loc_end_clifford = self.clifford_algebra.embed(loc_end, (1, 2, 3))

        return full_embedding, loc_end_clifford, attention_mask

    def preprocess(self, batch):
        loc, vel, edge_attr, charges, loc_end, edges = batch
        loc_mean = self.compute_mean_centered(loc)
        loc_mean, vel, charges, loc_end = self.flatten_tensors(loc_mean, vel, charges, loc_end)
        return loc_mean, vel, edge_attr, charges, loc_end, edges

    def compute_mean_centered(self, tensor):
        return tensor - tensor.mean(dim=1, keepdim=True)

    def flatten_tensors(self, *tensors):
        return [tensor.float().view(-1, *tensor.shape[2:]) for tensor in tensors]

    def get_node_embedding(self, loc_mean, vel, charges):
        # Embed data in Clifford space
        invariants = self.clifford_algebra.embed(charges, (0,))
        xv = torch.stack([loc_mean, vel], dim=1)
        covariants = self.clifford_algebra.embed(xv, (1, 2, 3))
        # stack the node info
        nodes_stack = torch.cat([invariants[:, None], covariants], dim=1)
        # project the node embeddings to higher dimensions
        full_node_embedding = self.node_projection(nodes_stack)
        return full_node_embedding, nodes_stack

    def get_edge_embedding(self, edge_attr, edges, batch_size, n_nodes, nodes_stack):
        # Get edge nodes and edge features
        # edges are batch_size, 2, 20 - number of edges per graph
        edges = self.get_edge_nodes(edges, batch_size, n_nodes)
        #edge_attr = edge_attr[:, indices, :]  # batch_size, 20, 1 ->  batch_size, 10, 1 gets rid of duplicate edges
        full_edge_embedding = self.get_full_edge_embedding(edge_attr, nodes_stack, edges)
        return full_edge_embedding, edges

    def get_edge_nodes(self, edges, batch_size, n_nodes):
        # Edge nodes start off as [batch_size, 2, 20] need to be reduced to [batch_size, 2, 10]
        batch_index = torch.arange(batch_size, device=edges.device)
        edges = edges + n_nodes * batch_index[:, None, None]  # [batch_size, 2, 20]

        #edges, indices = self.get_unique_edges_with_indices(edges[0])

        #edges = edges.unsqueeze(0).repeat(batch_size,1,1) # [batch_size, 2, 10]
        edges = tuple(edges.transpose(0, 1).flatten(1))
        return edges #, indices

    def get_unique_edges_with_indices(self, tensor):
        edges = set()  # edges before
        unique_edges = []
        unique_indices = []

        for i, edge in enumerate(tensor.t()):
            node1, node2 = sorted(edge.tolist())
            if (node1, node2) not in edges:
                edges.add((node1, node2))
                unique_edges.append((node1, node2))
                unique_indices.append(i)

        unique_edges_tensor = torch.tensor(unique_edges).t()
        unique_indices_tensor = torch.tensor(unique_indices)

        return unique_edges_tensor, unique_indices_tensor

    def get_full_edge_embedding(self, edge_attr, nodes_in_clifford, edges):

        orig_edge_attr_clifford = self.clifford_algebra.embed(edge_attr[..., None], (0,)).view(-1, 1, 8)

        extra_edge_attr_clifford = self.make_edge_attr(nodes_in_clifford, edges)  # should be [batch
        edge_attr_all = torch.cat((orig_edge_attr_clifford, extra_edge_attr_clifford), dim=1)
        # Project the edge features to higher dimensions
        projected_edges = self.edge_projection(edge_attr_all)

        return projected_edges

    def make_edge_attr(self, node_features, edges):

        node1_features = node_features[edges[0]]
        node2_features = node_features[edges[1]]
        # difference = node1_features - node2_features
        gp = self.clifford_algebra.geometric_product(node1_features, node2_features)
        edge_attributes = torch.cat((node1_features, node2_features, gp), dim=1)
        return edge_attributes

    def get_attention_mask(self, batch_size, n_nodes, edges):
        num_edges_per_graph = edges[0].size(0) // batch_size

        # Initialize an attention mask with zeros for a single batch
        base_attention_mask = torch.zeros(1,  num_edges_per_graph+n_nodes, num_edges_per_graph+n_nodes, device=edges[0].device)

        # Nodes can attend to themselves and to all other nodes within the same graph
        for i in range(n_nodes):
            for j in range(n_nodes):
                base_attention_mask[0, i, j] = 1

        for i in range(num_edges_per_graph):
            start_node = edges[0][i].item()
            end_node = edges[1][i].item()
            edge_idx = n_nodes + i

            # Edges can attend to their corresponding nodes
            base_attention_mask[0, edge_idx, start_node] = 1
            base_attention_mask[0, edge_idx, end_node] = 1

            # Nodes can attend to their corresponding edges
            base_attention_mask[0, start_node, edge_idx] = 1
            base_attention_mask[0, end_node, edge_idx] = 1

        # Stack the masks for each batch
        attention_mask = base_attention_mask.repeat(batch_size, 1, 1)

        # Convert the mask to float and set masked positions to -inf and allowed positions to 0
        attention_mask = attention_mask.float()
        attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attention_mask.masked_fill(attention_mask == 1, float(0.0))

        # Set the diagonal of the attention mask to 0
        attention_mask[0].fill_diagonal_(float('-inf'))

        return attention_mask

    # def get_attention_mask(self,batch_size, n_nodes, edges):
    #     num_edges_per_graph = edges[0].size(0) // batch_size
    #
    #     # Initialize an attention mask with zeros for a single batch
    #     base_attention_mask = torch.zeros(1, 25, 25, device=edges[0].device)
    #
    #     for i in range(num_edges_per_graph):
    #         start_node = edges[0][i].item()
    #         end_node = edges[1][i].item()
    #         edge_idx = n_nodes + i
    #
    #         # Edges can attend to their corresponding nodes
    #         base_attention_mask[0, edge_idx, start_node] = 1
    #         base_attention_mask[0, edge_idx, end_node] = 1
    #
    #         # Nodes can attend to their corresponding edges
    #         base_attention_mask[0, start_node, edge_idx] = 1
    #         base_attention_mask[0, end_node, edge_idx] = 1
    #
    #     # Stack the masks for each batch
    #     attention_mask = base_attention_mask.repeat(batch_size, 1, 1)
    #
    #     # Convert the mask to float and set masked positions to -inf and allowed positions to 0
    #     attention_mask = attention_mask.float()
    #     attention_mask.masked_fill(attention_mask == 0, float('-inf'))
    #     attention_mask.masked_fill(attention_mask == 1, float(0.0))
    #
    #     return attention_mask
