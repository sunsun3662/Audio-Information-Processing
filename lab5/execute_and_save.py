"""
Execute lab5 notebooks and save outputs directly to the notebook files.
Uses run_lab5.py and run_lab5_exp3.py results + captures stdout for each cell.
"""
import os, sys, warnings, json, io, traceback
warnings.filterwarnings('ignore')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# Apply patches
import speechbrain.utils.importutils as _iu
_orig = _iu.LazyModule.ensure_module
def _patched(self, stacklevel=1):
    try:
        return _orig(self, stacklevel)
    except ImportError:
        name = str(self)
        if 'k2_fsa' in name or 'nlp' in name or 'wordemb' in name:
            import types
            mod = types.ModuleType(name.split('target=')[-1].split(',')[0].strip() if 'target=' in name else 'dummy')
            sys.modules[mod.__name__] = mod
            return mod
        raise
_iu.LazyModule.ensure_module = _patched

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def execute_cell(source_code):
    """Execute code and capture stdout/stderr."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    outputs = []
    try:
        exec(source_code, {'__name__': '__main__', '__file__': '<notebook>'})
        stdout_val = sys.stdout.getvalue()
        stderr_val = sys.stderr.getvalue()
        if stdout_val:
            outputs.append({"output_type": "stream", "name": "stdout", "text": stdout_val.splitlines(True)})
        if stderr_val:
            outputs.append({"output_type": "stream", "name": "stderr", "text": stderr_val.splitlines(True)})
    except Exception as e:
        tb = traceback.format_exc()
        outputs.append({
            "output_type": "error",
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": tb.splitlines(True)
        })
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return outputs

def process_notebook(nb_path):
    """Execute all code cells in a notebook and save outputs."""
    print(f"\n{'='*60}")
    print(f"Processing {nb_path}")
    print(f"{'='*60}")

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Build execution context that persists across cells
    exec_globals = {'__name__': '__main__', '__file__': nb_path}

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if not source.strip():
            continue

        print(f"  Executing cell [{i}]...", end=' ')
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        outputs = []

        try:
            exec(source, exec_globals)
            stdout_val = sys.stdout.getvalue()
            stderr_val = sys.stderr.getvalue()
            if stdout_val:
                outputs.append({"output_type": "stream", "name": "stdout", "text": stdout_val.splitlines(True)})
            if stderr_val:
                outputs.append({"output_type": "stream", "name": "stderr", "text": stderr_val.splitlines(True)})
            print("OK")
        except Exception as e:
            tb = traceback.format_exc()
            outputs.append({
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": tb.splitlines(True)
            })
            print(f"ERROR: {e}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        cell['outputs'] = outputs

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  Saved {nb_path}")

# Execute notebooks
for nb in ['01_setup.ipynb', '02_speaker_embedding.ipynb', '03_asv_asi.ipynb']:
    process_notebook(nb)

print("\nAll notebooks executed and saved!")
