# Olive Service

[![CI](https://github.com/ahsanazmi1/olive/workflows/CI/badge.svg)](https://github.com/ahsanazmi1/olive/actions/workflows/ci.yml)
[![Contracts](https://github.com/ahsanazmi1/olive/workflows/Contracts/badge.svg)](https://github.com/ahsanazmi1/olive/actions/workflows/contracts.yml)
[![Security](https://github.com/ahsanazmi1/olive/workflows/Security/badge.svg)](https://github.com/ahsanazmi1/olive/actions/workflows/security.yml)

**Olive** is the **Loyalty and Incentives service** for the [Open Checkout Network (OCN)](https://github.com/ahsanazmi1/ocn-common).

## Phase 2 — Explainability

🚧 **Currently in development** - Phase 2 focuses on AI-powered explainability and human-readable loyalty decision reasoning.

- **Status**: Active development on `phase-2-explainability` branch
- **Features**: LLM integration, explainability API endpoints, decision audit trails
- **Issue Tracker**: [Phase 2 Issues](https://github.com/ahsanazmi1/olive/issues?q=is%3Aopen+is%3Aissue+label%3Aphase-2)
- **Timeline**: Weeks 4-8 of OCN development roadmap

Olive provides intelligent loyalty programs and incentive management for the OCN ecosystem. Unlike traditional black-box loyalty systems, Olive offers:

## Quickstart (≤ 60s)

Get up and running with Olive OCN Agent in under a minute:

```bash
# Clone the repository
git clone https://github.com/ahsanazmi1/olive.git
cd olive

# Setup everything (venv, deps, pre-commit hooks)
make setup

# Run tests to verify everything works
make test

# Start the service
make run
```

**That's it!** 🎉

The service will be running at `http://localhost:8000`. Test the endpoints:

```bash
# Health check
curl http://localhost:8000/health

# MCP getStatus
curl -X POST http://localhost:8000/mcp/invoke \
  -H "Content-Type: application/json" \
  -d '{"verb": "getStatus", "args": {}}'

# MCP listIncentives
curl -X POST http://localhost:8000/mcp/invoke \
  -H "Content-Type: application/json" \
  -d '{"verb": "listIncentives", "args": {}}'
```

### Additional Makefile Targets

```bash
make lint        # Run code quality checks
make fmt         # Format code with black/ruff
make clean       # Remove virtual environment and cache
make help        # Show all available targets
```

## Manual Setup (Alternative)

If you prefer manual setup over the Makefile:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
pytest -q

# Start the service
uvicorn olive.api:app --reload
```

## API Endpoints

### Core Endpoints
- `GET /health` - Health check endpoint

### MCP (Model Context Protocol)
- `POST /mcp/invoke` - MCP protocol endpoint for Olive service operations
  - `getStatus` - Get the current status of the Olive agent
  - `listIncentives` - List available incentives and rewards in the OCN ecosystem

## Development

This project uses:
- **FastAPI** for the web framework
- **pytest** for testing
- **ruff** and **black** for code formatting
- **mypy** for type checking
- **pre-commit** for code quality hooks

### Pre-commit Hooks

Install and run pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install the hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

The hooks include:
- **ruff** - Fast Python linter and formatter
- **black** - Code formatting
- **end-of-file-fixer** - Ensures files end with newlines
- **trailing-whitespace** - Removes trailing whitespace
- **mypy** - Type checking
- **bandit** - Security linting
- **yaml/json** validation

## Phase 3 — Negotiation & Live Fee Bidding

Merchant policy DSL & routing knobs.

### Phase 3 — Negotiation & Live Fee Bidding
- [x] Policy DSL for routing (discounts, loyalty, early-pay)
- [x] MCP verbs to configure/retrieve policies
- [x] Tests for policy enforcement in negotiation flows

## License

MIT License - see [LICENSE](LICENSE) file for details.
