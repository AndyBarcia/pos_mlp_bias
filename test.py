import unittest
import torch

from functions import PosMLP, PosMLPAttention, PairPosMLP, PosMLPSelfAttention


class TestPosMLP(unittest.TestCase):
    def test_non_batched(self):
        model = PosMLP(dim=32, batched=False).to('cuda')
        pos = torch.rand(10, 4, device='cuda')
        output = model(pos, size=(16, 16))
        self.assertEqual(output.shape, (10, 16, 16))

    def test_batched(self):
        model = PosMLP(dim=32, batched=True).to('cuda')
        pos = torch.rand(10, 4, device='cuda')
        queries = torch.rand(10, 32, device='cuda')
        output = model(pos, size=16, queries=queries)
        self.assertEqual(output.shape, (10, 16, 16))

    def test_multi_dim_input_batched(self):
        model = PosMLP(dim=32, batched=True).to('cuda')
        pos = torch.rand(2, 5, 4, device='cuda')
        queries = torch.rand(2, 5, 32, device='cuda')
        output = model(pos, size=(16, 16), queries=queries)
        self.assertEqual(output.shape, (2, 5, 16, 16))

    def test_implementations(self):
        model = PosMLP(dim=32, batched=True).to('cuda')
        pos = torch.rand(10, 4, device='cuda')
        queries = torch.rand(10, 32, device='cuda')
        # Test that both 'cuda' and 'python' code paths execute without error
        output_cuda = model(pos, size=16, queries=queries, implementation='cuda')
        self.assertEqual(output_cuda.shape, (10, 16, 16))
        output_python = model(pos, size=16, queries=queries, implementation='python')
        self.assertEqual(output_python.shape, (10, 16, 16))


class TestPairPosMLP(unittest.TestCase):
    def test_non_batched(self):
        model = PairPosMLP(dim=32, batched=False).to('cuda')
        pos1 = torch.rand(10, 5, 4, device='cuda')
        pos2 = torch.rand(10, 7, 4, device='cuda')
        output = model(pos1, pos2)
        self.assertEqual(output.shape, (10, 5, 7))

    def test_batched(self):
        model = PairPosMLP(dim=32, batched=True).to('cuda')
        pos1 = torch.rand(10, 5, 4, device='cuda')
        pos2 = torch.rand(10, 7, 4, device='cuda')
        queries = torch.rand(10, 5, 32, device='cuda')
        output = model(pos1, pos2, queries=queries)
        self.assertEqual(output.shape, (10, 5, 7))

    def test_multi_dim_input_batched(self):
        model = PairPosMLP(dim=32, batched=True).to('cuda')
        pos1 = torch.rand(2, 3, 5, 4, device='cuda')
        pos2 = torch.rand(2, 3, 7, 4, device='cuda')
        queries = torch.rand(2, 3, 5, 32, device='cuda')
        output = model(pos1, pos2, queries=queries)
        self.assertEqual(output.shape, (2, 3, 5, 7))

    def test_implementations(self):
        model = PairPosMLP(dim=32, batched=True).to('cuda')
        pos1 = torch.rand(10, 5, 4, device='cuda')
        pos2 = torch.rand(10, 7, 4, device='cuda')
        queries = torch.rand(10, 5, 32, device='cuda')
        # Test that both 'cuda' and 'python' code paths execute without error
        output_cuda = model(pos1, pos2, queries=queries, implementation='cuda')
        self.assertEqual(output_cuda.shape, (10, 5, 7))
        output_python = model(pos1, pos2, queries=queries, implementation='python')
        self.assertEqual(output_python.shape, (10, 5, 7))


class TestPosMLPAttention(unittest.TestCase):
    def test_forward(self):
        dim, n_heads, H, W, B = 32, 8, 16, 16, 4
        model = PosMLPAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, 4, device='cuda')
        output = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, dim))

    def test_multi_dim_input(self):
        dim, n_heads, H, W, B, N = 32, 8, 16, 16, 4, 3
        model = PosMLPAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, N, dim, device='cuda')
        memory = torch.rand(B, N, H, W, dim, device='cuda')
        pos = torch.rand(B, N, 4, device='cuda')
        output = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, N, dim))

    def test_different_k_dim(self):
        dim, k_dim, n_heads, H, W, B = 32, 64, 8, 16, 16, 4
        model = PosMLPAttention(dim=dim, k_dim=k_dim, n_heads=n_heads).to('cuda')
        queries = torch.rand(B, dim, device='cuda')
        memory = torch.rand(B, H, W, k_dim, device='cuda')
        pos = torch.rand(B, 4, device='cuda')
        output = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, dim))


class TestPosMLPSelfAttention(unittest.TestCase):
    def test_forward(self):
        dim, n_heads, q_len, B = 32, 8, 12, 4
        model = PosMLPSelfAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        x = torch.rand(B, q_len, dim, device='cuda')
        pos = torch.rand(B, q_len, 4, device='cuda')
        output = model(x, pos)
        self.assertEqual(output.shape, (B, q_len, dim))

    def test_multi_dim_input(self):
        dim, n_heads, q_len, B, N = 32, 8, 12, 4, 3
        model = PosMLPSelfAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        x = torch.rand(B, N, q_len, dim, device='cuda')
        pos = torch.rand(B, N, q_len, 4, device='cuda')
        output = model(x, pos)
        self.assertEqual(output.shape, (B, N, q_len, dim))

if __name__ == '__main__':
    unittest.main()