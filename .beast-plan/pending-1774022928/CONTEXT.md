# Beast-Plan Context — hackernews-simulator OSS Launch

## Task Description
Transform ~/code/hn-simulator into a polished open-source tool "hackernews-simulator" ready for Show HN launch. Maximum virality — the tool predicts its own HN reception (meta hook).

## Codebase Summary
- Working project: 256 tests, 20+ source modules, Python 3.12
- LightGBM (regression + multiclass 5-class) + LanceDB RAG + Claude CLI
- 144K stories, 319K comments indexed
- CLI: predict, compare, suggest-loop, backtest commands
- v2 features: SHAP explanations, percentile calibration, time-of-day advice, domain reputation

## Decisions
1. **Project name:** hackernews-simulator (rename from hn-simulator)
2. **GitHub:** Personal account (malakhov)
3. **Distribution:** `pip install git+https://github.com/malakhov/hackernews-simulator` (no PyPI)
4. **Pre-trained model:** Include artifacts in GitHub release assets or HuggingFace model hub
5. **Streamlit UI:** Full-featured (compare mode, suggest loop, history)
6. **Self-optimization:** Run tool on itself, include prediction in README as meta demo
7. **Demo data:** Include small pre-trained model for instant usage

## Scope
### In Scope
- Rename project hn-simulator → hackernews-simulator throughout
- Professional README with demo output, badges, installation, examples
- LICENSE (MIT), CONTRIBUTING.md
- GitHub Actions CI (pytest on push/PR)
- Polish CLI output (rich/click colors, progress bars, better formatting)
- `hn-sim init` one-command setup (download pre-trained artifacts OR train from scratch)
- Pre-trained model artifacts (upload to HuggingFace or GitHub releases)
- Streamlit web UI (full: predict, compare, suggest-loop, history)
- Self-optimization: generate best Show HN title using the tool itself
- README examples with actual impressive outputs

### Out of Scope
- PyPI publish (just git install)
- Mobile app
- Hosted web service
- Fine-tuning local LLM

## Constraints
- Existing 256 tests must pass after rename
- Claude CLI for LLM calls (no API key requirement for end users too — document this)
- M4 Pro 24GB development machine
- Pre-trained artifacts must be downloadable without auth (public repo/HF)
