import unittest
import torch

from functions import (
    PosMLP, PosMLPAttention, 
    PosGaussian, PosGaussianAttention,
    PairPosMLP, PosMLPSelfAttention,
    PosPairGaussian, PosPairGaussianAttention
)


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

    def test_dynamic_kernel(self):
        model = PosMLP(dim=32, batched=True).to('cuda')
        pos = torch.rand(10, 4, device='cuda')
        queries = torch.rand(10, 32, device='cuda')
        output = model(pos, size=(13,12), queries=queries)
        self.assertEqual(output.shape, (10, 13, 12))

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

class TestPosGaussian(unittest.TestCase):
    def test_batched(self):
        model = PosGaussian(dim=32, n_heads=4).to('cuda')
        pos = torch.rand(10, 4, device='cuda')
        queries = torch.rand(10, 32, device='cuda')
        output = model(pos, size=16, queries=queries)
        self.assertEqual(output.shape, (10, 4, 16, 16))

    def test_dynamic_kernel(self):
        model = PosGaussian(dim=32, n_heads=4).to('cuda')
        pos = torch.rand(10, 4, device='cuda')
        queries = torch.rand(10, 32, device='cuda')
        output = model(pos, size=(13,12), queries=queries)
        self.assertEqual(output.shape, (10, 4, 13, 12))

    def test_multi_dim_input_batched(self):
        model = PosGaussian(dim=32, n_heads=4).to('cuda')
        pos = torch.rand(2, 5, 4, device='cuda')
        queries = torch.rand(2, 5, 32, device='cuda')
        output = model(pos, size=(16, 16), queries=queries)
        self.assertEqual(output.shape, (2, 5, 4, 16, 16))

    def test_implementations(self):
        model = PosGaussian(dim=32, n_heads=4).to('cuda')
        pos = torch.rand(10, 4, device='cuda')
        queries = torch.rand(10, 32, device='cuda')
        # Test that both 'cuda' and 'python' code paths execute without error
        output_cuda = model(pos, size=16, queries=queries, implementation='cuda')
        self.assertEqual(output_cuda.shape, (10, 4, 16, 16))
        output_python = model(pos, size=16, queries=queries, implementation='python')
        self.assertEqual(output_python.shape, (10, 4, 16, 16))

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
        dim, n_heads, Q, H, W, B = 32, 8, 10, 16, 16, 4
        model = PosMLPAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, Q, dim))

    def test_dynamic_forward(self):
        dim, n_heads, Q, H, W, B = 32, 8, 10, 24, 13, 4
        model = PosMLPAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, Q, dim))

    def test_multi_dim_input(self):
        dim, n_heads, Q, H, W, B, N = 32, 8, 10, 16, 16, 4, 3
        model = PosMLPAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, N, Q, dim, device='cuda')
        memory = torch.rand(B, N, H, W, dim, device='cuda')
        pos = torch.rand(B, N, Q, 4, device='cuda')
        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, N, Q, dim))

    def test_different_k_dim(self):
        dim, k_dim, n_heads, Q, H, W, B = 32, 64, 8, 10, 16, 16, 4
        model = PosMLPAttention(dim=dim, k_dim=k_dim, n_heads=n_heads).to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, k_dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, Q, dim))
    
    def test_pos_embd(self):
        dim, n_heads, Q, H, W, B = 32, 8, 10, 16, 16, 4
        model = PosMLPAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        queries_pos_emb = torch.rand(B, Q, dim, device='cuda')
        memory_pos_emb = torch.rand(B, H, W, dim, device='cuda')
        output, _ = model(queries, memory, pos, query_pos_emb=queries_pos_emb, memory_pos_emb=memory_pos_emb)
        self.assertEqual(output.shape, (B, Q, dim))

    def test_return_attn_logits(self):
        dim, n_heads, Q, H, W, B = 32, 8, 10, 16, 16, 4
        model = PosMLPAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        output, logits = model(queries, memory, pos, return_attn_logits=True)
        self.assertEqual(output.shape, (B, Q, dim))
        self.assertEqual(logits.shape, (B, n_heads, Q, H * W))
    
    def test_attn_mask(self):
        dim, n_heads, Q, H, W, B = 32, 8, 10, 16, 16, 4
        model = PosMLPAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        attn_mask = torch.randint(0, 2, (B*n_heads, Q, H*W), dtype=torch.bool, device='cuda')
        output, _ = model(queries, memory, pos, attn_mask=attn_mask)
        self.assertEqual(output.shape, (B, Q, dim))


class TestPosGaussianAttention(unittest.TestCase):
    def test_forward(self):
        dim, n_heads, Q, H, W, B = 32, 4, 10, 16, 16, 4
        model = PosGaussianAttention(dim=dim, n_heads=n_heads).to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, Q, dim))

    def test_dynamic_forward(self):
        dim, n_heads, Q, H, W, B = 32, 4, 10, 24, 13, 4
        model = PosGaussianAttention(dim=dim, n_heads=n_heads).to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, Q, dim))

    def test_multi_dim_input(self):
        dim, n_heads, Q, H, W, B, N = 32, 4, 10, 16, 16, 4, 3
        model = PosGaussianAttention(dim=dim, n_heads=n_heads).to('cuda')
        queries = torch.rand(B, N, Q, dim, device='cuda')
        memory = torch.rand(B, N, H, W, dim, device='cuda')
        pos = torch.rand(B, N, Q, 4, device='cuda')
        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, N, Q, dim))

    def test_different_k_dim(self):
        dim, k_dim, n_heads, Q, H, W, B = 32, 64, 4, 10, 16, 16, 4
        model = PosGaussianAttention(dim=dim, k_dim=k_dim,n_heads=n_heads).to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, k_dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, Q, dim))


class TestPosMLPSelfAttention(unittest.TestCase):
    def test_forward(self):
        dim, n_heads, Q, B = 32, 8, 12, 4
        model = PosMLPSelfAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        x = torch.rand(B, Q, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        output, _ = model(x, pos)
        self.assertEqual(output.shape, (B, Q, dim))

    def test_multi_dim_input(self):
        dim, n_heads, Q, B, N = 32, 8, 12, 4, 3
        model = PosMLPSelfAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        x = torch.rand(B, N, Q, dim, device='cuda')
        pos = torch.rand(B, N, Q, 4, device='cuda')
        output, _ = model(x, pos)
        self.assertEqual(output.shape, (B, N, Q, dim))
    
    def test_pos_embd(self):
        dim, n_heads, Q, B = 32, 8, 12, 4
        model = PosMLPSelfAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        x = torch.rand(B, Q, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        pos_emb = torch.rand(B, Q, dim, device='cuda')
        output, _ = model(x, pos, pos_emb=pos_emb)
        self.assertEqual(output.shape, (B, Q, dim))

class TestPosPairGaussian(unittest.TestCase):
    def test_forward_shape(self):
        dim, n_heads, N1, N2, B = 32, 8, 10, 12, 4
        model = PosPairGaussian(dim=dim, n_heads=n_heads, implementation="cuda").to('cuda')
        pos_queries = torch.rand(B, N1, 4, device='cuda')
        pos_keys = torch.rand(B, N2, 4, device='cuda')
        queries = torch.rand(B, N1, dim, device='cuda')
        keys = torch.rand(B, N2, dim, device='cuda')
        
        output = model(pos_queries, pos_keys, queries, keys)
        self.assertEqual(output.shape, (B, n_heads, N1, N2))

    def test_multi_dim_input(self):
        dim, n_heads, N1, N2, B, N = 32, 8, 10, 12, 4, 3
        model = PosPairGaussian(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        pos_queries = torch.rand(B, N, N1, 4, device='cuda')
        pos_keys = torch.rand(B, N, N2, 4, device='cuda')
        queries = torch.rand(B, N, N1, dim, device='cuda')
        keys = torch.rand(B, N, N2, dim, device='cuda')

        output = model(pos_queries, pos_keys, queries, keys)
        self.assertEqual(output.shape, (B, N, n_heads, N1, N2))

    def test_implementations(self):
        dim, n_heads, N1, N2, B = 32, 8, 10, 12, 4
        model = PosPairGaussian(dim=dim, n_heads=n_heads).to('cuda')
        pos_queries = torch.rand(B, N1, 4, device='cuda')
        pos_keys = torch.rand(B, N2, 4, device='cuda')
        queries = torch.rand(B, N1, dim, device='cuda')
        keys = torch.rand(B, N2, dim, device='cuda')

        # Test python implementation
        output_python = model(pos_queries, pos_keys, queries, keys, implementation='python')
        self.assertEqual(output_python.shape, (B, n_heads, N1, N2))

        # Test CUDA implementation
        output_cuda = model(pos_queries, pos_keys, queries, keys, implementation='cuda')
        self.assertEqual(output_cuda.shape, (B, n_heads, N1, N2))
        
    def test_learned_scale(self):
        dim, n_heads, N1, N2, B = 32, 8, 10, 12, 4
        model = PosPairGaussian(dim=dim, n_heads=n_heads, implementation="python", learned_scale=True).to('cuda')
        pos_queries = torch.rand(B, N1, 4, device='cuda')
        pos_keys = torch.rand(B, N2, 4, device='cuda')
        queries = torch.rand(B, N1, dim, device='cuda')
        keys = torch.rand(B, N2, dim, device='cuda')
        
        output = model(pos_queries, pos_keys, queries, keys)
        self.assertEqual(output.shape, (B, n_heads, N1, N2))
        self.assertTrue(hasattr(model, 'scales'))

    def test_normalize(self):
        dim, n_heads, N1, N2, B = 32, 8, 10, 12, 4
        model = PosPairGaussian(dim=dim, n_heads=n_heads, implementation="python", normalize=True).to('cuda')
        pos_queries = torch.rand(B, N1, 4, device='cuda')
        pos_keys = torch.rand(B, N2, 4, device='cuda')
        queries = torch.rand(B, N1, dim, device='cuda')
        keys = torch.rand(B, N2, dim, device='cuda')
        
        output = model(pos_queries, pos_keys, queries, keys)
        self.assertEqual(output.shape, (B, n_heads, N1, N2))
        self.assertTrue(hasattr(model, 'q_offset_norm'))
        self.assertTrue(hasattr(model, 'q_sigma_norm'))
        self.assertTrue(hasattr(model, 'k_offset_norm'))
        self.assertTrue(hasattr(model, 'k_sigma_norm'))


class TestPosPairGaussianAttention(unittest.TestCase):
    def test_forward_shape(self):
        dim, n_heads, Q, H, W, B = 32, 8, 10, 16, 16, 4
        model = PosPairGaussianAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        
        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, Q, dim))

    def test_dynamic_spatial_shape(self):
        dim, n_heads, Q, H, W, B = 32, 8, 10, 24, 13, 4
        model = PosPairGaussianAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        
        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, Q, dim))

    def test_multi_dim_input(self):
        dim, n_heads, Q, H, W, B, N = 32, 8, 10, 16, 16, 4, 3
        model = PosPairGaussianAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, N, Q, dim, device='cuda')
        memory = torch.rand(B, N, H, W, dim, device='cuda')
        pos = torch.rand(B, N, Q, 4, device='cuda')
        
        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, N, Q, dim))

    def test_different_k_dim(self):
        dim, k_dim, n_heads, Q, H, W, B = 32, 64, 8, 10, 16, 16, 4
        model = PosPairGaussianAttention(dim=dim, k_dim=k_dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, k_dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')

        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, Q, dim))
        
    def test_with_pos_embed(self):
        dim, n_heads, Q, H, W, B = 32, 8, 10, 16, 16, 4
        model = PosPairGaussianAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        query_pos_emb = torch.rand(B, Q, dim, device='cuda')
        memory_pos_emb = torch.rand(B, H, W, dim, device='cuda')

        output, _ = model(queries, memory, pos, query_pos_emb=query_pos_emb, memory_pos_emb=memory_pos_emb)
        self.assertEqual(output.shape, (B, Q, dim))

    def test_return_attn_logits(self):
        dim, n_heads, Q, H, W, B = 32, 8, 10, 16, 16, 4
        model = PosPairGaussianAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        
        output, logits = model(queries, memory, pos, return_attn_logits=True)
        self.assertEqual(output.shape, (B, Q, dim))
        self.assertIsNotNone(logits)
        self.assertEqual(logits.shape, (B, n_heads, Q, H * W))
    
    def test_attn_mask(self):
        dim, n_heads, Q, H, W, B = 32, 8, 10, 16, 16, 4
        model = PosPairGaussianAttention(dim=dim, n_heads=n_heads, implementation="python").to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')
        attn_mask = torch.randint(0, 2, (B, n_heads, Q, H, W), dtype=torch.bool, device='cuda')
        
        output, _ = model(queries, memory, pos, attn_mask=attn_mask)
        self.assertEqual(output.shape, (B, Q, dim))
        
    def test_only_gaussian_logits(self):
        dim, n_heads, Q, H, W, B = 32, 8, 10, 16, 16, 4
        model = PosPairGaussianAttention(dim=dim, n_heads=n_heads, implementation="python", only_gaussian_logits=True).to('cuda')
        queries = torch.rand(B, Q, dim, device='cuda')
        memory = torch.rand(B, H, W, dim, device='cuda')
        pos = torch.rand(B, Q, 4, device='cuda')

        self.assertFalse(hasattr(model, 'q_proj'))
        self.assertFalse(hasattr(model, 'kv_proj'))
        self.assertTrue(hasattr(model, 'v_proj'))
        
        output, _ = model(queries, memory, pos)
        self.assertEqual(output.shape, (B, Q, dim))

if __name__ == '__main__':
    unittest.main()