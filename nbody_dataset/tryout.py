import torch

# Original implementation
def original_attention_mask(batch_size, n_nodes, edges, device):
    batch_index = torch.arange(batch_size, device=device)
    edges = edges + n_nodes * batch_index[:, None, None]
    edges = tuple(edges.transpose(0, 1).flatten(1))
    start_nodes, end_nodes = edges

    attention_mask = torch.zeros(batch_size * (n_nodes + 20), batch_size * (n_nodes + 20), device=device)

    for b in range(batch_size):
        node_start_idx = b * (n_nodes + 20)
        edge_start_idx = node_start_idx + n_nodes

        for i in range(20):
            start_node = start_nodes[i + b * 20].item() + node_start_idx
            end_node = end_nodes[i + b * 20].item() + node_start_idx
            edge_idx = edge_start_idx + i

            attention_mask[edge_idx, start_node] = 1
            attention_mask[edge_idx, end_node] = 1
            attention_mask[start_node, edge_idx] = 1
            attention_mask[end_node, edge_idx] = 1

    attention_mask = attention_mask.float()
    attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
    attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))

    return attention_mask

# Optimized implementation
def optimized_attention_mask(batch_size, n_nodes, edges, device):
    num_edges_per_graph = edges.size(1) // batch_size
    total_elements_per_graph = n_nodes + num_edges_per_graph

    attention_mask = torch.full((batch_size * total_elements_per_graph, batch_size * total_elements_per_graph), float('-inf'), device=device)

    batch_index = torch.arange(batch_size, device=device).view(-1, 1, 1)
    node_start_indices = batch_index * total_elements_per_graph
    edge_start_indices = node_start_indices + n_nodes

    start_nodes = edges[0] + node_start_indices
    end_nodes = edges[1] + node_start_indices
    edge_indices = torch.arange(num_edges_per_graph, device=device).view(1, -1) + edge_start_indices

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

    return attention_mask

# Test function to compare the outputs
def test_attention_masks():
    batch_size = 3
    n_nodes = 5
    num_edges = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example edges tensor
    edges = torch.randint(0, n_nodes, (2, batch_size * num_edges), device=device)

    original_mask = original_attention_mask(batch_size, n_nodes, edges, device)
    optimized_mask = optimized_attention_mask(batch_size, n_nodes, edges, device)

    # Compare the results
    if torch.allclose(original_mask, optimized_mask):
        print("The attention masks are equivalent.")
    else:
        print("The attention masks are not the same!")
        print("Original Mask:")
        print(original_mask)
        print("Optimized Mask:")
        print(optimized_mask)
        print("Difference:")
        print(original_mask - optimized_mask)

# Run the test
test_attention_masks()
