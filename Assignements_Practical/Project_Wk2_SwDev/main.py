from fastapi import FastAPI

import customers, dashboard
from logger import setup_logger

logger = setup_logger(__name__)

app = FastAPI(
    title="Classic Models API",
    description=(
        "A layered FastAPI application built on the Classic Models dataset."
    ),
    version="1.0.0",
)

# Register routers
# Dashboard router first so /customers/count is matched before /customers/{id}
app.include_router(dashboard.router)
app.include_router(customers.router)

logger.info("FastAPI application started.")


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Classic Models API is running.",
        "docs": "/docs",
        "redoc": "/redoc",
    }
