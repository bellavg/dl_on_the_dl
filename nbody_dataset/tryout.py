import torch
import torch.nn as nn
from algebra.cliffordalgebra import CliffordAlgebra
from dataset import NBody

metric = [1, 1, 1]
clifford_algebra = CliffordAlgebra(metric)


def test_equivariant_dropout():
    dropout_prob = 0.5
    layer = EquivariantDropout(p=dropout_prob)
    layer.train()  # Set to training mode

    input_data = torch.randn(10, 25, 16,
                             8)  # Batch size 10, (num_edges + num_nodes) 25, hidden_dim 16, Clifford space 8
    output = layer(input_data)

    # Check that the output shape matches the input shape
    assert output.shape == input_data.shape, "Output shape does not match input shape"

    # Check dropout functionality
    num_elements = input_data.numel()
    num_dropped = (output == 0).sum().item()
    expected_dropped = num_elements * dropout_prob

    print(f"Number of elements: {num_elements}")
    print(f"Number of dropped elements: {num_dropped}")
    print(f"Expected dropped elements: {expected_dropped}")

    # Allow for some statistical variance
    assert abs(num_dropped - expected_dropped) < 0.1 * expected_dropped, "Unexpected number of elements dropped"


def test_equivariant_properties():
    dropout_prob = 0.5
    layer = EquivariantDropout(p=dropout_prob)
    layer.train()  # Set to training mode

    algebra = CliffordAlgebra([1, 1, 1])
    input_data = torch.randn(10, 25, 16,
                             8)  # Batch size 10, (num_edges + num_nodes) 25, hidden_dim 16, Clifford space 8

    # Apply dropout
    output_data = layer(input_data)

    # Check that the norms are maintained
    input_norm = algebra.norm(input_data)
    output_norm = algebra.norm(output_data)

    print("Input Norm:", input_norm)
    print("Output Norm:", output_norm)

    # Tolerance for numerical precision
    tolerance = 1e-5  # Increased tolerance
    assert torch.allclose(input_norm, output_norm, atol=tolerance), "Equivariance not maintained after dropout"


class EquivariantDropout(nn.Module):
    """
    Equivariant Dropout layer for multivectors.

    Parameters
    ----------
    p : float
        Dropout probability.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Applies dropout.

        Parameters
        ----------
        x : torch.Tensor with shape (batch, num_edges + num_nodes, hidden_dim, 8)
            Multivector inputs.

        Returns
        -------
        torch.Tensor with shape (batch, num_edges + num_nodes, hidden_dim, 8)
            Multivector inputs with dropout applied.
        """
        if not self.training or self.p == 0.0:
            return x

        # Generate dropout mask
        mask_shape = x[..., :1].shape
        mask = torch.bernoulli((1 - self.p) * torch.ones(mask_shape, device=x.device))
        mask = mask / (1 - self.p)  # Scale the mask to keep the expected value the same

        # Apply the mask to the input tensor
        output = x * mask

        return output


def test_gradient_flow():
    dropout_prob = 0.5
    layer = EquivariantDropout(p=dropout_prob)
    layer.train()  # Set to training mode

    input_data = torch.randn(10, 25, 16, 8, requires_grad=True)  # Enable gradient tracking
    output = layer(input_data)

    # Perform a simple reduction to create a scalar loss
    loss = output.mean()
    loss.backward()

    # Check if gradients are computed for the input
    assert input_data.grad is not None, "Gradients not flowing through dropout layer"

    # Check that gradients are not zero (unless dropout probability is 1.0)
    if dropout_prob < 1.0:
        assert input_data.grad.abs().sum().item() > 0, "Gradients are zero after dropout"


# Run the tests
test_equivariant_dropout()
test_equivariant_properties()
test_gradient_flow()
