import torch
import torch.nn as nn
import pytest
from toolkit.util.quantize import quantize, get_qtype

def test_torchao_quantization_types():
    """
    Test that all torchao quantization types (uint2 to uint8, float8)
    can be successfully applied to a model's linear layers.
    """
    qtypes = ["uint2", "uint3", "uint4", "uint5", "uint6", "uint7", "uint8", "float8"]
    
    for qtype_name in qtypes:
        # Create a simple Linear layer
        model = nn.Linear(32, 32)
        
        # Get the corresponding qtype
        weights_qtype = get_qtype(qtype_name)
        
        # Quantize the model
        quantize(model, weights=weights_qtype)
        
        # Verify that the weight has been quantized and is no longer a standard FloatTensor
        weight_class_name = type(model.weight).__name__
        assert "Tensor" in weight_class_name, f"Expected quantized weight tensor, got {weight_class_name}"
        assert weight_class_name != "Parameter", f"Weight should be quantized for {qtype_name}, but is still a Parameter"
