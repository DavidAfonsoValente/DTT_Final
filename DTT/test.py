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
    expected_file = os.path.join(vendored_path, 'transformers', 'models', 'gpt2', 'modeling_gpt2.py')
    print(f"Expected file exists: {os.path.exists(expected_file)}")
    
    # Import and reload full package
    try:
        import transformers
        importlib.reload(transformers)
        print(f"Transformers loaded from: {transformers.__file__}")
    except Exception as e:
        print(f"Transformers import error: {e}")
        sys.exit(1)
    
    try:
        from transformers.models import gpt2
        importlib.reload(gpt2)
        print(f"GPT2 module from: {gpt2.__file__}")
    except Exception as e:
        print(f"GPT2 import error: {e}")
        sys.exit(1)
    
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model
        module_file = getattr(GPT2Model.__module__, '__file__', 'No file attribute')
        print(f"GPT2Model loaded from module: {GPT2Model.__module__}")
        print(f"GPT2Model file: {module_file}")
    except Exception as e:
        print(f"GPT2Model import error: {e}")
        sys.exit(1)

# Test load
print("Loading custom GPT-2...")
try:
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained('gpt2')
    print(f"Has blend_gate_r: {hasattr(m.transformer, 'blend_gate_r')}")
    print(f"Has blend method: {hasattr(m.transformer, 'blend')}")
    print(f"Has blend_lambda: {hasattr(m.transformer, 'blend_lambda')}")
    print("Test passed!" if hasattr(m.transformer, 'blend_gate_r') else "Test failed - check mods in file")
except Exception as e:
    print(f"Model load error: {e}")