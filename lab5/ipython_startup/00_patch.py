import sys, warnings
warnings.filterwarnings('ignore')
import speechbrain.utils.importutils as _iu
_orig = _iu.LazyModule.ensure_module
def _patched(self, stacklevel=1):
    try:
        return _orig(self, stacklevel)
    except ImportError:
        name = str(self)
        if 'k2_fsa' in name or 'nlp' in name or 'wordemb' in name:
            import types
            target = self.target if hasattr(self, 'target') else 'dummy'
            mod = types.ModuleType(target)
            sys.modules[target] = mod
            return mod
        raise
_iu.LazyModule.ensure_module = _patched
