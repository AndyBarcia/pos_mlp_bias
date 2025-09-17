import torch
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import psutil
import os
import sys
from contextlib import contextmanager

from functions import (
    BoxRPBCUDAFunction, box_rbp_python,
    BoxBRPBCUDAFunction, box_brbp_python,
    BoxBMHRPBCUDAFunction, box_bmhrbp_python,
    BoxPairRPBCUDAFunction, box_pair_rbp_python,
    BoxPairBRPBCUDAFunction, box_pair_brbp_python,
    AttentionCUDAFunction, attn_python,
)

@contextmanager
def _suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


class CUDAKernelTester:
    """
    A class to generalize the testing of CUDA kernels against a Python implementation.
    It benchmarks performance and memory, checks for correctness, and presents
    the results in a comparison table.
    """

    def __init__(self, cuda_function, python_function, input_creators, arg_order, device="cuda", dtype=torch.float32):
        """
        Initializes the tester.

        Args:
            cuda_function: The CUDA kernel function to be tested (e.g., MyFunc.apply).
            python_function: The reference Python implementation of the kernel.
            input_creators: A dictionary of lambda functions to generate input tensors.
            arg_order (list[str]): A list of argument names in the exact order the function expects them.
            device (str, optional): The device to run the tests on. Defaults to "cuda".
            dtype (torch.dtype, optional): The data type for the tensors. Defaults to torch.float32.
        """
        self.cuda_function = cuda_function
        self.python_function = python_function
        self.input_creators = input_creators
        self.arg_order = arg_order
        self.device = torch.device(device)
        self.dtype = dtype

    def _measure_pass(self, func, *args):
        """
        Profiles a function to get total CUDA time and peak memory usage.
        
        This function sums the CUDA time of ALL kernels launched during the
        profiler's context, providing an accurate measurement for asynchronous operations.
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        # Suppress verbose profiler logs
        with _suppress_stdout_stderr():
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                output = func(*args)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        
        # Sum up all CUDA time from all events in the profile
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        total_cuda_time_ms = sum([e.cuda_time for e in prof.key_averages()]) / 1000.0
        peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == 'cuda' else 0
        
        return output, total_cuda_time_ms, peak_mem_mb

    def _check_correctness(self, tensor_cuda, tensor_python, label, atol, rtol):
        """Checks if two tensors are close and prints detailed differences if they are not."""
        are_close = torch.allclose(tensor_cuda, tensor_python.to(self.device), atol=atol, rtol=rtol)
        if not are_close:
            print(f"\n--- Correctness FAILED for '{label}' ---")
            abs_diff = (tensor_cuda - tensor_python.to(self.device)).abs()
            rel_diff = abs_diff / (tensor_python.to(self.device).abs() + 1e-8)
            print(f"  Max Absolute Difference: {abs_diff.max().item():.6e}")
            print(f"  Mean Absolute Difference: {abs_diff.mean().item():.6e}")
            print(f"  Max Relative Difference: {rel_diff.max().item():.6e}")
            print("-" * 40)
        return are_close

    def run(self, fwd_atol=1e-5, fwd_rtol=1e-4, bwd_atol=1e-4, bwd_rtol=1e-3, **kwargs):
        """
        Executes the full testing pipeline: performance, memory, and correctness.
        """
        function_name = self.cuda_function.__self__.__name__
        
        print(f"\n{'='*80}")
        print(f"Starting test for {function_name}")
        print(f"{'='*80}")

        # Initialize results dictionary
        results = {
            'pytorch': {'fwd_time_ms': 0.0, 'bwd_time_ms': 0.0, 'fwd_mem_mb': 0.0, 'bwd_mem_mb': 0.0},
            'cuda':    {'fwd_time_ms': 0.0, 'bwd_time_ms': 0.0, 'fwd_mem_mb': 0.0, 'bwd_mem_mb': 0.0},
            'correctness': {'fwd': False, 'bwd': True}
        }

        # Create base input tensors once
        base_inputs = {name: creator(self.device, self.dtype) for name, creator in self.input_creators.items()}
        
        # --- Python Implementation ---
        inputs_py = {name: t.clone().requires_grad_(True) for name, t in base_inputs.items()}
        all_args_py = {**inputs_py, **kwargs}
        ordered_args_py = [all_args_py[name] for name in self.arg_order]
        
        output_py, fwd_time_py, fwd_mem_py = self._measure_pass(self.python_function, *ordered_args_py)
        results['pytorch']['fwd_time_ms'] = fwd_time_py
        results['pytorch']['fwd_mem_mb'] = fwd_mem_py
        loss_py = output_py.sum()

        _, bwd_time_py, bwd_mem_py = self._measure_pass(lambda: loss_py.backward())
        results['pytorch']['bwd_time_ms'] = bwd_time_py
        results['pytorch']['bwd_mem_mb'] = max(fwd_mem_py, bwd_mem_py) # Peak memory is the max of fwd and bwd

        # --- CUDA Implementation ---
        inputs_cu = {name: t.clone().requires_grad_(True) for name, t in base_inputs.items()}
        all_args_cu = {**inputs_cu, **kwargs}
        ordered_args_cu = [all_args_cu[name] for name in self.arg_order]
        
        output_cu, fwd_time_cu, fwd_mem_cu = self._measure_pass(self.cuda_function, *ordered_args_cu)
        results['cuda']['fwd_time_ms'] = fwd_time_cu
        results['cuda']['fwd_mem_mb'] = fwd_mem_cu
        loss_cu = output_cu.sum()
        
        _, bwd_time_cu, bwd_mem_cu = self._measure_pass(lambda: loss_cu.backward())
        results['cuda']['bwd_time_ms'] = bwd_time_cu
        results['cuda']['bwd_mem_mb'] = max(fwd_mem_cu, bwd_mem_cu)
        
        # --- Correctness Checks ---
        results['correctness']['fwd'] = self._check_correctness(output_cu, output_py, "Forward Pass Output", fwd_atol, fwd_rtol)
        
        for name in base_inputs:
            if inputs_cu[name].grad is not None and inputs_py[name].grad is not None:
                is_grad_correct = self._check_correctness(inputs_cu[name].grad, inputs_py[name].grad, f"Gradient of '{name}'", bwd_atol, bwd_rtol)
                results['correctness']['bwd'] &= is_grad_correct

        # Define column widths
        pass_w, impl_w, time_w, mem_w, corr_w = 10, 14, 15, 20, 10
        
        # Create header and separator strings based on widths
        header = f"| {'Pass':<{pass_w}} | {'Implementation':<{impl_w}} | {'Time (ms)':>{time_w}} | {'Peak GPU Mem (MB)':>{mem_w}} | {'Correct?':>{corr_w}} |"
        separator = f"|{'-'*(pass_w+2)}|{'-'*(impl_w+2)}|{'-'*(time_w+2)}|{'-'*(mem_w+2)}|{'-'*(corr_w+2)}|"
        
        print(separator)
        print(header)
        print(separator)
        
        fwd_correct = 'PASS' if results['correctness']['fwd'] else 'FAIL'
        bwd_correct = 'PASS' if results['correctness']['bwd'] else 'FAIL'

        # Forward Pass Data
        print(f"| {'Forward':<{pass_w}} | {'Pytorch':<{impl_w}} | {results['pytorch']['fwd_time_ms']:>{time_w}.4f} | {results['pytorch']['fwd_mem_mb']:>{mem_w}.2f} | {'':>{corr_w}} |")
        print(f"| {'Forward':<{pass_w}} | {'CUDA':<{impl_w}} | {results['cuda']['fwd_time_ms']:>{time_w}.4f} | {results['cuda']['fwd_mem_mb']:>{mem_w}.2f} | {fwd_correct:>{corr_w}} |")
        print(separator)
        # Backward Pass Data
        print(f"| {'Backward':<{pass_w}} | {'Pytorch':<{impl_w}} | {results['pytorch']['bwd_time_ms']:>{time_w}.4f} | {results['pytorch']['bwd_mem_mb']:>{mem_w}.2f} | {'':>{corr_w}} |")
        print(f"| {'Backward':<{pass_w}} | {'CUDA':<{impl_w}} | {results['cuda']['bwd_time_ms']:>{time_w}.4f} | {results['cuda']['bwd_mem_mb']:>{mem_w}.2f} | {bwd_correct:>{corr_w}} |")
        print(separator)


def test_box_rbp():
    B, Ch = 16 * 300, 16
    H, W = 64, 64

    input_creators = {
        "mlp_weights": lambda device, dtype: torch.randn(4 * Ch + 1, device=device, dtype=dtype),
        "pos": lambda device, dtype: torch.cat([
            torch.rand(B, 2, device=device, dtype=dtype),
            torch.rand(B, 2, device=device, dtype=dtype) * 0.5 + 0.1
        ], dim=-1),
    }
    arg_order = ["mlp_weights", "pos", "Ch", "W", "H"]

    tester = CUDAKernelTester(
        cuda_function=BoxRPBCUDAFunction.apply,
        python_function=box_rbp_python,
        input_creators=input_creators,
        arg_order=arg_order
    )
    tester.run(Ch=Ch, W=W, H=H)


def test_box_brbp():
    B, Ch = 16 * 300, 16
    H, W = 128, 128

    input_creators = {
        "mlp_weights": lambda device, dtype: torch.randn(B, 4 * Ch + 1, device=device, dtype=dtype),
        "pos": lambda device, dtype: torch.cat([
            torch.rand(B, 2, device=device, dtype=dtype),
            torch.rand(B, 2, device=device, dtype=dtype) * 0.5 + 0.1
        ], dim=-1),
    }
    arg_order = ["mlp_weights", "pos", "Ch", "W", "H"]

    tester = CUDAKernelTester(
        cuda_function=BoxBRPBCUDAFunction.apply,
        python_function=box_brbp_python,
        input_creators=input_creators,
        arg_order=arg_order
    )    
    tester.run(Ch=Ch, W=W, H=H)


def test_box_bmhrbp():
    B, Ch = 16*300, 16
    H, W = 64, 64
    Nh = 8

    input_creators = {
        "mlp_weights": lambda device, dtype: torch.randn(B, 2 * Ch + Ch + Ch * Nh + Nh, device=device, dtype=dtype),
        "pos": lambda device, dtype: torch.cat([
            torch.rand(B, 2, device=device, dtype=dtype),
            torch.rand(B, 2, device=device, dtype=dtype) * 0.5 + 0.1
        ], dim=-1),
    }
    arg_order = ["mlp_weights", "pos", "Ch", "Nh", "H", "W"]

    tester = CUDAKernelTester(
        cuda_function=BoxBMHRPBCUDAFunction.apply,
        python_function=box_bmhrbp_python,
        input_creators=input_creators,
        arg_order=arg_order
    )    
    tester.run(Ch=Ch, Nh=Nh, W=W, H=H)


def test_box_pair_rbp():
    B, N1, N2, Ch = 16, 300, 300, 16

    input_creators = {
        "mlp_weights": lambda device, dtype: torch.randn(6 * Ch + 1, device=device, dtype=dtype),
        "pos1": lambda device, dtype: torch.cat([
            torch.rand(B, N1, 2, device=device, dtype=dtype),
            torch.rand(B, N1, 2, device=device, dtype=dtype) * 0.5 + 0.1
        ], dim=-1),
        "pos2": lambda device, dtype: torch.cat([
            torch.rand(B, N2, 2, device=device, dtype=dtype),
            torch.rand(B, N2, 2, device=device, dtype=dtype) * 0.5 + 0.1
        ], dim=-1),
    }
    arg_order = ["mlp_weights", "pos1", "pos2", "Ch"]

    tester = CUDAKernelTester(
        cuda_function=BoxPairRPBCUDAFunction.apply,
        python_function=box_pair_rbp_python,
        input_creators=input_creators,
        arg_order=arg_order
    )    
    tester.run(Ch=Ch)


def test_box_pair_brbp():
    B, N1, N2, Ch = 16, 300, 300, 16

    input_creators = {
        "mlp_weights": lambda device, dtype: torch.randn(B, N1, 6 * Ch + 1, device=device, dtype=dtype),
        "pos1": lambda device, dtype: torch.cat([
            torch.rand(B, N1, 2, device=device, dtype=dtype),
            torch.rand(B, N1, 2, device=device, dtype=dtype) * 0.5 + 0.1
        ], dim=-1),
        "pos2": lambda device, dtype: torch.cat([
            torch.rand(B, N2, 2, device=device, dtype=dtype),
            torch.rand(B, N2, 2, device=device, dtype=dtype) * 0.5 + 0.1
        ], dim=-1),
    }
    arg_order = ["mlp_weights", "pos1", "pos2", "Ch"]

    tester = CUDAKernelTester(
        cuda_function=BoxPairBRPBCUDAFunction.apply,
        python_function=box_pair_brbp_python,
        input_creators=input_creators,
        arg_order=arg_order
    )    
    tester.run(Ch=Ch)


def test_attention():
    B, Nh, Nq, Nk, C = 16, 8, 300, 300, 32

    input_creators = {
        "q": lambda device, dtype: torch.randn(B, Nh, Nq, C, device=device, dtype=dtype),
        "k": lambda device, dtype: torch.randn(B, Nh, Nk, C, device=device, dtype=dtype),
        "v": lambda device, dtype: torch.randn(B, Nh, Nk, C, device=device, dtype=dtype),
    }
    arg_order = ["q", "k", "v"]

    tester = CUDAKernelTester(
        cuda_function=AttentionCUDAFunction.apply,
        python_function=attn_python,
        input_creators=input_creators,
        arg_order=arg_order
    )    
    tester.run()


if __name__ == "__main__":
    #test_attention()
    #test_box_rbp()
    #test_box_brbp()
    #test_box_pair_rbp()
    #test_box_pair_brbp()
    test_box_bmhrbp()