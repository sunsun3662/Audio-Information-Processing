"""
Execute all lab5 notebooks with k2_fsa patch applied.
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# Apply k2_fsa patch BEFORE importing speechbrain
import speechbrain.utils.importutils as _iu
_orig = _iu.LazyModule.ensure_module
def _patched(self, stacklevel=1):
    try:
        return _orig(self, stacklevel)
    except ImportError:
        if 'k2_fsa' in str(self):
            import types
            mod = types.ModuleType('speechbrain.integrations.k2_fsa')
            sys.modules['speechbrain.integrations.k2_fsa'] = mod
            return mod
        raise
_iu.LazyModule.ensure_module = _patched

# Now execute notebooks
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat

os.chdir(os.path.dirname(os.path.abspath(__file__)))

notebooks = [
    ('01_setup.ipynb', 300),
    ('02_speaker_embedding.ipynb', 600),
    ('03_asv_asi.ipynb', 600),
]

for nb_name, timeout in notebooks:
    print(f"\n{'='*60}")
    print(f"Executing {nb_name}...")
    print(f"{'='*60}")

    with open(nb_name, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
    ep.allow_errors = True

    try:
        ep.preprocess(nb, {'metadata': {'path': '.'}})
        print(f"  {nb_name} executed successfully")
    except Exception as e:
        print(f"  {nb_name} error: {e}")

    # Save
    with open(nb_name, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"  Saved {nb_name}")

print("\nAll notebooks executed!")
