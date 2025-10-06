# Olive Service

[![CI](https://github.com/ahsanazmi1/olive/workflows/CI/badge.svg)](https://github.com/ahsanazmi1/olive/actions/workflows/ci.yml)
[![Contracts](https://github.com/ahsanazmi1/olive/workflows/Contracts/badge.svg)](https://github.com/ahsanazmi1/olive/actions/workflows/contracts.yml)
[![Security](https://github.com/ahsanazmi1/olive/workflows/Security/badge.svg)](https://github.com/ahsanazmi1/olive/actions/workflows/security.yml)

**Olive** is the **Loyalty and Incentives service** for the [Open Checkout Network (OCN)](https://github.com/ahsanazmi1/ocn-common).

## Phase 4 — Payment Instruction & Visibility

🚧 **Currently in development** - Phase 4 focuses on payment instruction generation, settlement visibility, and comprehensive payment tracking for loyalty operations.

- **Status**: Active development on `phase-4-instruction` branch
- **Features**: Payment instruction schemas, settlement visibility, payment tracking, instruction validation
- **Issue Tracker**: [Phase 4 Issues](https://github.com/ahsanazmi1/olive/issues?q=is%3Aopen+is%3Aissue+label%3Aphase-4)
- **Timeline**: Weeks 12-16 of OCN development roadmap

See [CHANGELOG.md](CHANGELOG.md) for detailed Phase 4 progress and features.

### Enhanced Policy DSL

Olive Phase 4 introduces a comprehensive Policy DSL for merchant routing rules, enabling sophisticated policy enforcement during Orca/Opal negotiations.

#### DSL Syntax

```yaml
policy_id: "unique_policy_identifier"
policy_name: "Human-readable policy name"
description: "Policy description"

# Legacy fields (backward compatible)
prefer_rail: "ACH" | "DEBIT" | "CREDIT" | "BNPL" | "STABLECOIN" | "PREPAID"
loyalty_rebate_pct: 0.0-100.0
early_pay_discount_bps: 0.0-10000.0

# Enhanced policy rules
conditions:
  - condition_type: "amount_range" | "time_window" | "merchant_category" | "customer_segment" | "payment_frequency" | "risk_level"
    field: "field_name"
    operator: "eq" | "gt" | "lt" | "gte" | "lte" | "in" | "between"
    value: any_value
    description: "Condition description"

actions:
  - action_type: "rail_preference" | "rebate_application" | "discount_application" | "loyalty_boost" | "tax_validation" | "early_pay_incentive"
    parameters:
      key: value
    weight: 0.0-10.0
    description: "Action description"

# Merchant routing rules
rebate_rules:
  RAIL_TYPE: percentage_value
early_pay_rules:
  RAIL_TYPE: basis_points_value
loyalty_incentives:
  incentive_type: multiplier_value
tax_validation_rules:
  required: true/false
  threshold: amount_value
  jurisdictions: ["US", "CA"]

# Policy enforcement
enforcement_mode: "advisory" | "mandatory" | "override"
override_threshold: 0.0-1.0

# Metadata
merchant_id: "merchant_identifier"
priority: 1-10
enabled: true/false
effective_from: "ISO_datetime"
effective_until: "ISO_datetime"
```

#### MCP Verbs

Olive exposes the following MCP verbs for policy management:

- **`setPolicy`**: Create or update a policy
- **`getPolicy`**: Retrieve policies by ID or merchant
- **`evaluatePolicies`**: Evaluate policies against transaction context
- **`enforcePolicies`**: Enforce policies during negotiation

#### Sample Configurations

See `config/sample_policies.yaml` for comprehensive examples including:
- Debit preference policies with conditions
- Early payment bonus policies
- ACH preference with loyalty incentives
- Tax validation requirements
- Multi-condition complex policies

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
