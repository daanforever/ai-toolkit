import torch
import pytest
from toolkit.optimizers.optimizer_utils import copy_stochastic

def test_stochastic_rounding_float16_accumulation():
    """
    Test that stochastic rounding correctly accumulates small updates over time.
    Without stochastic rounding, updates smaller than 0.5 ULP are lost.
    With stochastic rounding, they should accumulate to the correct expected value.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Base value where ULP for float16 is 2^-15 (approx 3.05e-5)
    base_val = 1.0
    
    # Update size is 0.3 * ULP (smaller than 0.5 ULP, so standard rounding would lose it)
    update_val = 0.3 * (1/1024)
    
    # Create a large tensor to test the statistical mean
    num_elements = 100000
    source = torch.full((num_elements,), base_val + update_val, dtype=torch.float32, device=device)
    target = torch.zeros_like(source, dtype=torch.float16, device=device)
    
    # Apply stochastic rounding
    copy_stochastic(target, source)
    
    # Check that the mean of the rounded values is close to the expected value
    mean_val = target.float().mean().item()
    expected_val = base_val + update_val
    
    # The mean should be very close to the expected value (within a small tolerance)
    assert abs(mean_val - expected_val) < 1e-4, f"Expected {expected_val}, got {mean_val}"
    
    # Check that we actually have a mix of values (some rounded up, some rounded down)
    unique_vals = torch.unique(target)
    assert len(unique_vals) >= 2, "Stochastic rounding should produce a mix of rounded values"

def test_stochastic_rounding_float8_e4m3fn_accumulation():
    """Test stochastic rounding for float8_e4m3fn."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base_val = 1.0
    # For float8_e4m3fn, mantissa is 3 bits. ULP at 1.0 is 2^-3 = 0.125
    update_val = 0.3 * 0.125
    
    num_elements = 100000
    source = torch.full((num_elements,), base_val + update_val, dtype=torch.float32, device=device)
    target = torch.zeros_like(source, dtype=torch.float8_e4m3fn, device=device)
    
    copy_stochastic(target, source)
    
    mean_val = target.float().mean().item()
    expected_val = base_val + update_val
    
    assert abs(mean_val - expected_val) < 1e-2, f"Expected {expected_val}, got {mean_val}"
    assert len(torch.unique(target.float())) >= 2

def test_stochastic_rounding_float8_e5m2_accumulation():
    """Test stochastic rounding for float8_e5m2."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base_val = 1.0
    # For float8_e5m2, mantissa is 2 bits. ULP at 1.0 is 2^-2 = 0.25
    update_val = 0.3 * 0.25
    
    num_elements = 100000
    source = torch.full((num_elements,), base_val + update_val, dtype=torch.float32, device=device)
    target = torch.zeros_like(source, dtype=torch.float8_e5m2, device=device)
    
    copy_stochastic(target, source)
    
    mean_val = target.float().mean().item()
    expected_val = base_val + update_val
    
    assert abs(mean_val - expected_val) < 1e-2, f"Expected {expected_val}, got {mean_val}"
    assert len(torch.unique(target.float())) >= 2

if __name__ == "__main__":
    pytest.main([__file__])