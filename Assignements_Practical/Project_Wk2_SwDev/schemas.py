from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import List, Optional

from pydantic import BaseModel, validator

from logger import setup_logger

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Payment schemas
# ─────────────────────────────────────────────────────────────────────────────

class PaymentOut(BaseModel):
    customerNumber: int
    checkNumber: str
    paymentDate: date
    amount: Decimal

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────────────────────
# Order schemas
# ─────────────────────────────────────────────────────────────────────────────

class OrderOut(BaseModel):
    orderNumber: int
    orderDate: date
    requiredDate: date
    shippedDate: Optional[date] = None
    status: str
    comments: Optional[str] = None
    customerNumber: int

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────────────────────
# Customer schemas
# ─────────────────────────────────────────────────────────────────────────────

class CustomerCreate(BaseModel):
    """Used when creating a new customer (no ID required — DB assigns it)."""
    customerName: str
    contactLastName: str
    contactFirstName: str
    phone: str
    addressLine1: str
    addressLine2: Optional[str] = None
    city: str
    state: Optional[str] = None
    postalCode: Optional[str] = None
    country: str
    salesRepEmployeeNumber: Optional[int] = None
    creditLimit: Optional[Decimal] = None

    @validator("customerName")
    def name_not_empty(cls, v):
        if not v or not v.strip():
            logger.warning("Validation error: customerName is empty.")
            raise ValueError("customerName must not be empty.")
        return v

    @validator("creditLimit")
    def credit_limit_non_negative(cls, v):
        if v is not None and v < 0:
            logger.warning(f"Validation error: creditLimit {v} is negative.")
            raise ValueError("creditLimit must be >= 0.")
        return v


class CustomerUpdate(BaseModel):
    """
    All fields are Optional — a PATCH-style update.
    The caller supplies only the fields they want to change.
    """
    customerName: Optional[str] = None
    contactLastName: Optional[str] = None
    contactFirstName: Optional[str] = None
    phone: Optional[str] = None
    addressLine1: Optional[str] = None
    addressLine2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postalCode: Optional[str] = None
    country: Optional[str] = None
    salesRepEmployeeNumber: Optional[int] = None
    creditLimit: Optional[Decimal] = None

    @validator("creditLimit")
    def credit_limit_non_negative(cls, v):
        if v is not None and v < 0:
            logger.warning(f"Validation error: creditLimit {v} is negative.")
            raise ValueError("creditLimit must be >= 0.")
        return v


class CustomerOut(BaseModel):
    """What callers receive when they read a customer."""
    customerNumber: int
    customerName: str
    contactLastName: str
    contactFirstName: str
    phone: str
    addressLine1: str
    addressLine2: Optional[str] = None
    city: str
    state: Optional[str] = None
    postalCode: Optional[str] = None
    country: str
    salesRepEmployeeNumber: Optional[int] = None
    creditLimit: Optional[Decimal] = None

    class Config:
        from_attributes = True


class CustomerWithRelations(CustomerOut):
    """Extended view that includes related orders and payments."""
    orders: List[OrderOut] = []
    payments: List[PaymentOut] = []

    class Config:
        from_attributes = True


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard / count schema  (Task 3)
# ─────────────────────────────────────────────────────────────────────────────

class OverallCounts(BaseModel):
    customers: int
    orders: int
    products: int
    employees: int
    offices: int
    payments: int
    orderdetails: int
    productlines: int
