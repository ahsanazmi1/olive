# Olive Service

[![CI](https://github.com/ahsanazmi1/olive/workflows/CI/badge.svg)](https://github.com/ahsanazmi1/olive/actions/workflows/ci.yml)
[![Contracts](https://github.com/ahsanazmi1/olive/workflows/Contracts/badge.svg)](https://github.com/ahsanazmi1/olive/actions/workflows/contracts.yml)
[![Security](https://github.com/ahsanazmi1/olive/workflows/Security/badge.svg)](https://github.com/ahsanazmi1/olive/actions/workflows/security.yml)

Olive is a minimal Python service for the [Open Checkout Network (OCN)](https://github.com/ahsanazmi1/ocn-common). It provides core functionality and serves as a foundation for building more complex services within the OCN ecosystem. Olive follows modern Python development practices with FastAPI, comprehensive testing, and automated CI/CD workflows.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ahsanazmi1/olive.git
cd olive

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

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

## License

MIT License - see [LICENSE](LICENSE) file for details.
