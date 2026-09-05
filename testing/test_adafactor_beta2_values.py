import pytest
import torch
from toolkit.optimizers.adafactor import Adafactor

def run_adafactor_with_beta2(beta2, steps=20, lr=1e-4, constant_grad=False):
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.ones(8, 8))
    opt = Adafactor(
        [p],
        lr=lr,
        beta1=None,
        beta2=beta2,
        beta2_adaptive=False,
        scale_parameter=False,
        relative_step=False,
        weight_decay=0.0,
        clip_threshold=1.0,
        factored=False
    )
    
    updates = []
    for i in range(steps):
        if constant_grad:
            grad = torch.ones(8, 8) * 0.1
        else:
            # Varying gradient to avoid constant clipping
            grad = torch.ones(8, 8) * (0.1 / (i + 1))
            
        p.grad = grad.clone()
        before = p.detach().clone()
        opt.step()
        after = p.detach().clone()
        update = before - after
        updates.append(update.norm().item())
        
    return updates

def test_beta2_values_with_varying_grad():
    """
    Test that different beta2 values (0.9, 0.95, 0.99) produce different updates
    when gradients vary. This proves that beta2 is correctly applied.
    """
    updates_090 = run_adafactor_with_beta2(0.90, constant_grad=False)
    updates_095 = run_adafactor_with_beta2(0.95, constant_grad=False)
    updates_099 = run_adafactor_with_beta2(0.99, constant_grad=False)
    
    # After a few steps, the updates should diverge because the moving average
    # of squared gradients will adapt at different rates.
    assert updates_090[-1] != updates_095[-1]
    assert updates_095[-1] != updates_099[-1]
    assert updates_090[-1] != updates_099[-1]

def test_beta2_clipping_with_constant_grad():
    """
    Test that with a constant gradient, the updates might be identical for different
    beta2 values due to the clip_threshold. This explains why it might seem like
    beta2 has no effect.
    """
    updates_090 = run_adafactor_with_beta2(0.90, steps=5, constant_grad=True)
    updates_095 = run_adafactor_with_beta2(0.95, steps=5, constant_grad=True)
    updates_099 = run_adafactor_with_beta2(0.99, steps=5, constant_grad=True)
    
    # Initially, because exp_avg_sq is small, the update magnitude is large and gets
    # clipped to clip_threshold * lr. Thus, the updates are identical.
    for i in range(5):
        assert updates_090[i] == pytest.approx(updates_095[i], rel=1e-5)
        assert updates_095[i] == pytest.approx(updates_099[i], rel=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
