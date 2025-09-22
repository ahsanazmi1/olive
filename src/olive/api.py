"""
FastAPI application for Olive service.
"""

from typing import Any

from fastapi import FastAPI

import sys
from pathlib import Path

# Add project root to Python path for MCP imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp.server import mcp_router

# Create FastAPI application
app = FastAPI(
    title="Olive Service",
    description="Olive service for the Open Checkout Network (OCN)",
    version="0.1.0",
    contact={
        "name": "OCN Team",
        "email": "team@ocn.ai",
        "url": "https://github.com/ahsanazmi1/olive",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Include MCP router
app.include_router(mcp_router)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        dict: Health status information
    """
    return {"ok": True, "repo": "olive"}


def main() -> None:
    """Main entry point for running the application."""
    import uvicorn

    uvicorn.run(
        "olive.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
