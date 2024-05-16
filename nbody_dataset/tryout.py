import torch

class ExampleClass:
    def make_edge_attr(self, node_features, edges):
        node1_features = node_features[edges[0]]
        node2_features = node_features[edges[1]]
        difference = node1_features - node2_features
        edge_attributes = torch.cat((node1_features, node2_features, difference), dim=-1)
        return edge_attributes

def make_edge_attr(node_features, edges, batch_size=None):
    edge_attributes = []

    total_number_edges = edges[0].shape[0]

    # Loop over all edges
    for i in range(total_number_edges):
        node1 = edges[0][i]
        node2 = edges[1][i]

        # difference between node features
        node_i_features = node_features[node1]  # [#features(charge, loc, vel), dim]
        node_j_features = node_features[node2]  # [#features(charge, loc, vel), dim]
        difference = node_i_features - node_j_features
        edge_representation = torch.cat((node_i_features, node_j_features, difference), dim=-1)
        edge_attributes.append(edge_representation)

    edge_attributes = torch.stack(edge_attributes)
    return edge_attributes

# Define inputs
node_features = torch.rand((10, 4))  # 10 nodes, 4 features per node
edges = (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))  # Example edges

# Create instance of the class
example_class = ExampleClass()

# Call both functions
output1 = example_class.make_edge_attr(node_features, edges)
output2 = make_edge_attr(node_features, edges)

# Compare the outputs
print(torch.allclose(output1, output2))  # Should be True
print(output1)
print(output2)