import asyncio
import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

import crud
import schemas
from database import get_db
from logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(tags=["Dashboard"])


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — 8 individual count endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/customers/count")
def customers_count(db: Session = Depends(get_db)):
    logger.info("GET /customers/count — incoming request.")
    count = crud.get_customers_count(db)
    logger.info(f"GET /customers/count — response: {count}")
    return {"table": "customers", "count": count}


@router.get("/orders/count")
def orders_count(db: Session = Depends(get_db)):
    logger.info("GET /orders/count — incoming request.")
    count = crud.get_orders_count(db)
    logger.info(f"GET /orders/count — response: {count}")
    return {"table": "orders", "count": count}


@router.get("/products/count")
def products_count(db: Session = Depends(get_db)):
    logger.info("GET /products/count — incoming request.")
    count = crud.get_products_count(db)
    logger.info(f"GET /products/count — response: {count}")
    return {"table": "products", "count": count}


@router.get("/employees/count")
def employees_count(db: Session = Depends(get_db)):
    logger.info("GET /employees/count — incoming request.")
    count = crud.get_employees_count(db)
    logger.info(f"GET /employees/count — response: {count}")
    return {"table": "employees", "count": count}


@router.get("/offices/count")
def offices_count(db: Session = Depends(get_db)):
    logger.info("GET /offices/count — incoming request.")
    count = crud.get_offices_count(db)
    logger.info(f"GET /offices/count — response: {count}")
    return {"table": "offices", "count": count}


@router.get("/payments/count")
def payments_count(db: Session = Depends(get_db)):
    logger.info("GET /payments/count — incoming request.")
    count = crud.get_payments_count(db)
    logger.info(f"GET /payments/count — response: {count}")
    return {"table": "payments", "count": count}


@router.get("/orderdetails/count")
def orderdetails_count(db: Session = Depends(get_db)):
    logger.info("GET /orderdetails/count — incoming request.")
    count = crud.get_orderdetails_count(db)
    logger.info(f"GET /orderdetails/count — response: {count}")
    return {"table": "orderdetails", "count": count}


@router.get("/productlines/count")
def productlines_count(db: Session = Depends(get_db)):
    logger.info("GET /productlines/count — incoming request.")
    count = crud.get_productlines_count(db)
    logger.info(f"GET /productlines/count — response: {count}")
    return {"table": "productlines", "count": count}


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — Aggregated concurrent endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/overall_counts", response_model=schemas.OverallCounts)
async def overall_counts(db: Session = Depends(get_db)):
    """
    Returns row counts for all 8 tables in a single response.

    Concurrency design:
      • asyncio.gather() starts ALL 8 coroutines at once.
      • Each coroutine offloads its blocking SQLAlchemy call to a
        thread-pool executor via loop.run_in_executor().
      • Total response time ≈ slowest single query, not the sum.

    This is Factor VIII in action: concurrency over sequential blocking.
    """
    logger.info("GET /overall_counts — starting 8 concurrent DB queries.")
    start = time.perf_counter()

    (
        customers,
        orders,
        products,
        employees,
        offices,
        payments,
        orderdetails,
        productlines,
    ) = await asyncio.gather(
        crud.async_get_customers_count(db),
        crud.async_get_orders_count(db),
        crud.async_get_products_count(db),
        crud.async_get_employees_count(db),
        crud.async_get_offices_count(db),
        crud.async_get_payments_count(db),
        crud.async_get_orderdetails_count(db),
        crud.async_get_productlines_count(db),
    )

    elapsed = time.perf_counter() - start
    logger.info(
        f"GET /overall_counts — asyncio.gather() completed in {elapsed:.4f}s | "
        f"customers={customers}, orders={orders}, products={products}, "
        f"employees={employees}, offices={offices}, payments={payments}, "
        f"orderdetails={orderdetails}, productlines={productlines}"
    )

    return schemas.OverallCounts(
        customers=customers,
        orders=orders,
        products=products,
        employees=employees,
        offices=offices,
        payments=payments,
        orderdetails=orderdetails,
        productlines=productlines,
    )
