from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

import crud
import schemas
from database import get_db
from logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/customers", tags=["Customers"])


# ─────────────────────────────────────────────────────────────────────────────
# List — GET /customers/
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/", response_model=List[schemas.CustomerOut])
def list_customers(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """
    Returns a paginated list of customers.
    Use `skip` to offset and `limit` to control page size.
    """
    logger.info(f"GET /customers/ | skip={skip} limit={limit}")
    customers = crud.get_customers(db, skip=skip, limit=limit)
    logger.info(f"Returning {len(customers)} customer(s).")
    return customers


# ─────────────────────────────────────────────────────────────────────────────
# Create — POST /customers/
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/", response_model=schemas.CustomerOut, status_code=status.HTTP_201_CREATED)
def create_customer(
    customer: schemas.CustomerCreate,
    db: Session = Depends(get_db),
):
    """Creates a new customer. The database assigns the customerNumber."""
    logger.info(f"POST /customers/ | customerName={customer.customerName}")
    new_customer = crud.create_customer(db, customer)
    logger.info(f"Customer created successfully: customerNumber={new_customer.customerNumber}")
    return new_customer


# ─────────────────────────────────────────────────────────────────────────────
# Get one — GET /customers/{customer_number}
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{customer_number}", response_model=schemas.CustomerWithRelations)
def get_customer(
    customer_number: int,
    db: Session = Depends(get_db),
):
    """
    Returns a single customer along with their orders and payments.
    Returns 404 if the customer does not exist.
    """
    logger.info(f"GET /customers/{customer_number}")
    customer = crud.get_customer(db, customer_number)
    if customer is None:
        logger.warning(f"404 Not Found: customerNumber={customer_number}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_number} not found.",
        )
    return customer


# ─────────────────────────────────────────────────────────────────────────────
# Update — PUT /customers/{customer_number}
# ─────────────────────────────────────────────────────────────────────────────

@router.put("/{customer_number}", response_model=schemas.CustomerOut)
def update_customer(
    customer_number: int,
    customer_update: schemas.CustomerUpdate,
    db: Session = Depends(get_db),
):
    """
    Partially updates a customer (only supplied fields are changed).
    Returns 404 if the customer does not exist.
    """
    logger.info(f"PUT /customers/{customer_number}")
    updated = crud.update_customer(db, customer_number, customer_update)
    if updated is None:
        logger.warning(f"404 Not Found on update: customerNumber={customer_number}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_number} not found.",
        )
    logger.info(f"Customer updated: customerNumber={customer_number}")
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Delete — DELETE /customers/{customer_number}
# ─────────────────────────────────────────────────────────────────────────────

@router.delete("/{customer_number}", status_code=status.HTTP_204_NO_CONTENT)
def delete_customer(
    customer_number: int,
    db: Session = Depends(get_db),
):
    """
    Deletes a customer.  Returns 204 on success, 404 if not found.
    """
    logger.info(f"DELETE /customers/{customer_number}")
    success = crud.delete_customer(db, customer_number)
    if not success:
        logger.warning(f"404 Not Found on delete: customerNumber={customer_number}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_number} not found.",
        )
    logger.info(f"Customer deleted: customerNumber={customer_number}")


# ─────────────────────────────────────────────────────────────────────────────
# Related data — GET /customers/{id}/orders
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{customer_number}/orders", response_model=List[schemas.OrderOut])
def get_customer_orders(
    customer_number: int,
    db: Session = Depends(get_db),
):
    """
    Returns all orders for a given customer.
    Returns an empty list if the customer has no orders.
    Returns 404 if the customer does not exist.
    """
    logger.info(f"GET /customers/{customer_number}/orders")
    customer = crud.get_customer(db, customer_number)
    if customer is None:
        logger.warning(f"404 Not Found: customerNumber={customer_number}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_number} not found.",
        )
    orders = crud.get_customer_orders(db, customer_number)
    logger.info(f"Returning {len(orders)} order(s) for customerNumber={customer_number}")
    return orders


# ─────────────────────────────────────────────────────────────────────────────
# Related data — GET /customers/{id}/payments
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{customer_number}/payments", response_model=List[schemas.PaymentOut])
def get_customer_payments(
    customer_number: int,
    db: Session = Depends(get_db),
):
    """
    Returns all payments for a given customer.
    Returns an empty list if the customer has no payments.
    Returns 404 if the customer does not exist.
    """
    logger.info(f"GET /customers/{customer_number}/payments")
    customer = crud.get_customer(db, customer_number)
    if customer is None:
        logger.warning(f"404 Not Found: customerNumber={customer_number}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_number} not found.",
        )
    payments = crud.get_customer_payments(db, customer_number)
    logger.info(f"Returning {len(payments)} payment(s) for customerNumber={customer_number}")
    return payments
