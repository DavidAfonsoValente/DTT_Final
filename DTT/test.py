import sys
import os
import importlib

# Vendored path
project_root = os.path.dirname(os.path.abspath(__file__))
vendored_path = os.path.join(project_root, 'custom_transformers')
print(f"Project root: {project_root}")
print(f"Vendored path: {vendored_path}")
print(f"Vendored exists: {os.path.exists(vendored_path)}")

if os.path.exists(vendored_path):
    sys.path.insert(0, vendored_path)
    print(f"sys.path[0] after insert: {sys.path[0]}")
    
    # Check custom file
    expected_file = os.path.join(vendored_path, 'models', 'gpt2', 'modeling_gpt2.py')
    print(f"Expected file exists: {os.path.exists(expected_file)}")
    
    # Import and reload full package
    import transformers
    importlib.reload(transformers)
    print(f"Transformers loaded from: {transformers.__file__}")
    
    from transformers.models import gpt2
    importlib.reload(gpt2)
    print(f"GPT2 module from: {gpt2.__file__}")
    
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model
    print(f"GPT2Model loaded from: {GPT2Model.__module__.__file__}")

# Test load
print("Loading custom GPT-2...")
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('gpt2')
print(f"Has blend_gate_r: {hasattr(m.transformer, 'blend_gate_r')}")
print(f"Has blend method: {hasattr(m.transformer, 'blend')}")
print(f"Has blend_lambda: {hasattr(m.transformer, 'blend_lambda')}")
print("Test passed!" if hasattr(m.transformer, 'blend_gate_r') else "Test failed - check mods in file")