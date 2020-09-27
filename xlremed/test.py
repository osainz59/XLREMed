import torch
#import fire
import numpy as np

from .model import MultiGraphDecoder

def test_mgdecoder():
    print(f"Testing: {MultiGraphDecoder}")

    batch_size = 8
    seq_length = 3
    hidden_size = 768
    n_rel = 5

    # Generate input of shape (batch_size, seq_length, hidden_size)
    x = torch.randn((batch_size, seq_length, hidden_size))
    print("Input x:", x.size())

    layer = MultiGraphDecoder(n_rel, hidden_size).eval()
    A = layer(x)
    print(A)
    print(f"Correct output shape {(batch_size, n_rel, seq_length, seq_length)}: {A.shape == (batch_size, n_rel, seq_length, seq_length)}")
    print(f"Layer parameters {sum([np.prod(param.size()) for param in layer.parameters()])}")


if __name__ == "__main__":
    #fire.Fire(test_mgdecoder)
    test_mgdecoder()