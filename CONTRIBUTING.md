# Contributing to Olive

Thank you for your interest in contributing to Olive! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes
5. Run the test suite to ensure everything works
6. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/olive.git
cd olive

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest -q
```

## Code Style

This project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

Before submitting a pull request, ensure your code passes all checks:

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy .

# Run tests
pytest
```

## Testing

- Write tests for new functionality
- Ensure all tests pass: `pytest`
- Maintain or improve test coverage
- Use descriptive test names

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests if applicable
4. Ensure all CI checks pass
5. Submit a pull request with a clear description
6. Respond to feedback promptly

## Commit Messages

Use clear, descriptive commit messages following conventional commit format:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for test additions/changes

## Issues

- Use GitHub issues to report bugs or request features
- Provide clear descriptions and reproduction steps
- Include relevant system information

## License

By contributing to Olive, you agree that your contributions will be licensed under the MIT License.
