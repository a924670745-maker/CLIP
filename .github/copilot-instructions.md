## Quick orientation for coding agents

This repo is the OpenAI CLIP reference implementation in Python. The goal of these notes is to help an AI coding agent get productive quickly by describing the project's structure, patterns, and common developer flows with concrete examples.

Key files
- `clip/clip.py` — public entry points: `available_models()`, `load(name, device, jit, download_root)`, and `tokenize(...)`. This is the primary API surface consumers use.
- `clip/model.py` — core model implementation: `CLIP` class, vision/text backbones, `build_model(state_dict)` and `convert_weights()`.
- `clip/__init__.py` — convenience import of the public API.
- `setup.py`, `requirements.txt` — packaging and dependencies. `extras_require={'dev': ['pytest']}`.
- `tests/test_consistency.py` — canonical test: compares JIT vs non-JIT outputs across `clip.available_models()` and uses `CLIP.png` as a fixture.

Big-picture architecture
- Repo exposes a small, stable Python API surface in `clip/clip.py`. Most changes should preserve these functions' contracts (`load`, `tokenize`, `available_models`).
- Model loading has two paths:
  - JIT path: `load(..., jit=True)` uses `torch.jit.load` and performs graph-level patching to change device/dtype nodes.
  - State-dict path: `load(..., jit=False)` calls `build_model(state_dict)` in `clip/model.py` to reconstruct the Python model class.
- `build_model()` infers architecture from keys in a saved state dict (vision transformer vs modified resnet). When changing state dict keys or model shape, update this logic.
- Tokenization: `clip.tokenize()` uses a simple BPE-based tokenizer (`clip/simple_tokenizer.py`) and enforces a context length (default 77). Tokenizer encodings include start/end tokens; long inputs will raise unless `truncate=True`.

Developer workflows and commands
- Install dependencies and the package (recommended):

```powershell
pip install -r requirements.txt ; pip install -e .
```

- Run the example in README (Python usage example that calls `clip.load()` and `clip.tokenize()`).
- Run tests (note: tests will download models from the internet and may be large):

```powershell
pip install -e .[dev] ; pytest -q
```

- Model downloads: `load()` downloads into `~/.cache/clip` by default. You can override with `download_root` when calling `load(name, download_root=...)`.

Project-specific patterns & gotchas
- Model names are exact strings found in `_MODELS` in `clip/clip.py` (examples: `"ViT-B/32"`, `"RN50"`). Accepts either those names or a path to a local checkpoint file.
- JIT vs non-JIT parity: there is explicit test coverage ensuring logits match within tolerances (`tests/test_consistency.py`). When changing numerical routines or dtype handling, run this test.
- Device/dtype patching: JIT-loaded models are graph-patched to change device and dtype. If you change the JIT export format, review the patching logic in `clip/clip.py` (functions `patch_device`, `patch_float`).
- Preprocessing transform is returned by `load()` (second return value). Use that transform for images so input shapes and normalization match model expectations.
- Token length/context: CLIP models use context length 77. `tokenize()` will raise if input tokens > 77 unless `truncate=True`.

Integration points & external dependencies
- PyTorch (>=1.7.1 recommended) and torchvision are required; check `clip/clip.py` early version check.
- PIL is used for image IO; torchvision transforms are used for preprocessing.
- Models are downloaded from hard-coded URLs in `_MODELS` and validated via SHA256.

When editing or adding models
- Preserve state_dict key names the `build_model()` expects, or update `build_model()` accordingly.
- If adding a new architecture, add its name+URL to `_MODELS` and ensure `build_model()` and the JIT patching still work for the new artifact.

Examples to copy-paste
- Load a model (cpu, non-JIT) and tokenize:

```python
import clip
model, preprocess = clip.load('ViT-B/32', device='cpu', jit=False)
image = preprocess(Image.open('CLIP.png')).unsqueeze(0)
text = clip.tokenize(['a diagram', 'a dog']).to('cpu')
```

- Run the consistency test locally (downloads models):

```powershell
pytest tests/test_consistency.py -q
```

What not to assume
- There is no separate model registry; `_MODELS` in `clip/clip.py` is authoritative.
- Tests will download checkpoints — they are large. Avoid running the whole test suite on CI without caching or mocking downloads.

If anything below is unclear or you want more guidance (examples of common edits, how to add a new model, or CI/test recommendations), tell me which area to expand and I will iterate.
