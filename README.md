# TabFM-Stress

Zero-shot tabular foundation model benchmark for in-context learning (ICL) stress testing.

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install the project
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage

```bash
# Run a quick test (one model, one seed, minimal context)
uv run tabfm-stress run \
  --models tabpfn \
  --seeds 0 \
  --n-context 5 \
  --output results/test/

# Full benchmark (all models, default seeds and context sizes)
uv run tabfm-stress run \
  --models all \
  --seeds 0 1 2 \
  --n-context 5 13 26 40 47 \
  --output results/
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/...csv` | Path to dataset (.parquet or .csv) |
| `--models` | `all` | Comma-separated model names or `all` |
| `--seeds` | `0 1 2` | Random seeds |
| `--n-context` | `5 13 26 40 47` | Context subjects per run |
| `--feature-cols` | auto-detect | Comma-separated feature columns |
| `--target-col` | `label` | Target column name |
| `--subject-col` | `subject_id` | Subject ID column name |
| `--output` | `results/` | Output directory |

## Models

- **TabPFN** — `tabpfn`
- **TabDPT** — `tabdpt`
- **TabICL** — `tabicl`
- **Causilo** — `causilo`
- **TabFM** — `tabfm`

## Development

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Run pre-commit hooks
uv run pre-commit run --all-files
```
