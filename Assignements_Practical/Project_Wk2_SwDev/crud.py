from typing import List, Optional
import asyncio

from sqlalchemy.orm import Session

import models
import schemas
from logger import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Customer — CRUD
# ─────────────────────────────────────────────────────────────────────────────

def get_customers(db: Session, skip: int = 0, limit: int = 100) -> List[models.Customer]:
    logger.info(f"Fetching customers | skip={skip} limit={limit}")
    customers = db.query(models.Customer).offset(skip).limit(limit).all()
    logger.info(f"Fetched {len(customers)} customer(s).")
    return customers


def get_customer(db: Session, customer_number: int) -> Optional[models.Customer]:
    logger.info(f"Fetching customer | customerNumber={customer_number}")
    customer = (
        db.query(models.Customer)
        .filter(models.Customer.customerNumber == customer_number)
        .first()
    )
    if customer is None:
        logger.warning(f"Customer not found: customerNumber={customer_number}")
    else:
        logger.info(f"Customer found: customerNumber={customer_number}")
    return customer


def create_customer(db: Session, customer: schemas.CustomerCreate) -> models.Customer:
    logger.info(f"Creating customer: {customer.customerName}")
    db_customer = models.Customer(**customer.dict())
    db.add(db_customer)
    db.commit()
    db.refresh(db_customer)
    logger.info(f"Customer created: customerNumber={db_customer.customerNumber}")
    return db_customer


def update_customer(
    db: Session,
    customer_number: int,
    customer_update: schemas.CustomerUpdate,
) -> Optional[models.Customer]:
    logger.info(f"Updating customer: customerNumber={customer_number}")
    db_customer = get_customer(db, customer_number)
    if db_customer is None:
        return None

    update_data = customer_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_customer, field, value)

    db.commit()
    db.refresh(db_customer)
    logger.info(f"Customer updated: customerNumber={customer_number}")
    return db_customer


def delete_customer(db: Session, customer_number: int) -> bool:
    logger.info(f"Deleting customer: customerNumber={customer_number}")
    db_customer = get_customer(db, customer_number)
    if db_customer is None:
        logger.warning(f"Delete failed — customer not found: customerNumber={customer_number}")
        return False
    db.delete(db_customer)
    db.commit()
    logger.info(f"Customer deleted: customerNumber={customer_number}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Customer — related data
# ─────────────────────────────────────────────────────────────────────────────

def get_customer_orders(db: Session, customer_number: int) -> List[models.Order]:
    logger.info(f"Fetching orders for customerNumber={customer_number}")
    orders = (
        db.query(models.Order)
        .filter(models.Order.customerNumber == customer_number)
        .all()
    )
    logger.info(f"Found {len(orders)} order(s) for customerNumber={customer_number}")
    return orders


def get_customer_payments(db: Session, customer_number: int) -> List[models.Payment]:
    logger.info(f"Fetching payments for customerNumber={customer_number}")
    payments = (
        db.query(models.Payment)
        .filter(models.Payment.customerNumber == customer_number)
        .all()
    )
    logger.info(f"Found {len(payments)} payment(s) for customerNumber={customer_number}")
    return payments


# ─────────────────────────────────────────────────────────────────────────────
# Count functions — one per table (Task 3, Factor VIII)
# ─────────────────────────────────────────────────────────────────────────────

def get_customers_count(db: Session) -> int:
    logger.debug("Counting customers table.")
    count = db.query(models.Customer).count()
    logger.info(f"customers count = {count}")
    return count


def get_orders_count(db: Session) -> int:
    logger.debug("Counting orders table.")
    count = db.query(models.Order).count()
    logger.info(f"orders count = {count}")
    return count


def get_products_count(db: Session) -> int:
    logger.debug("Counting products table.")
    count = db.query(models.Product).count()
    logger.info(f"products count = {count}")
    return count


def get_employees_count(db: Session) -> int:
    logger.debug("Counting employees table.")
    count = db.query(models.Employee).count()
    logger.info(f"employees count = {count}")
    return count


def get_offices_count(db: Session) -> int:
    logger.debug("Counting offices table.")
    count = db.query(models.Office).count()
    logger.info(f"offices count = {count}")
    return count


def get_payments_count(db: Session) -> int:
    logger.debug("Counting payments table.")
    count = db.query(models.Payment).count()
    logger.info(f"payments count = {count}")
    return count


def get_orderdetails_count(db: Session) -> int:
    logger.debug("Counting orderdetails table.")
    count = db.query(models.OrderDetail).count()
    logger.info(f"orderdetails count = {count}")
    return count


def get_productlines_count(db: Session) -> int:
    logger.debug("Counting productlines table.")
    count = db.query(models.ProductLine).count()
    logger.info(f"productlines count = {count}")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Async wrappers — allow asyncio.gather() in the router (Task 3)
# ─────────────────────────────────────────────────────────────────────────────

async def async_get_customers_count(db: Session) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_customers_count, db)


async def async_get_orders_count(db: Session) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_orders_count, db)


async def async_get_products_count(db: Session) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_products_count, db)


async def async_get_employees_count(db: Session) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_employees_count, db)


async def async_get_offices_count(db: Session) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_offices_count, db)


async def async_get_payments_count(db: Session) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_payments_count, db)


async def async_get_orderdetails_count(db: Session) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_orderdetails_count, db)


async def async_get_productlines_count(db: Session) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_productlines_count, db)
