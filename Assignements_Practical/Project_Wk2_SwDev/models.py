from sqlalchemy import (
    Column, Integer, SmallInteger, String, Numeric,
    Date, Text, LargeBinary, ForeignKey,
)
from sqlalchemy.orm import relationship

from database import Base


class ProductLine(Base):
    __tablename__ = "productlines"

    productLine = Column(String(50), primary_key=True)
    textDescription = Column(String(4000), nullable=True)
    htmlDescription = Column(Text, nullable=True)
    image = Column(LargeBinary, nullable=True)

    products = relationship("Product", back_populates="product_line_rel")


class Product(Base):
    __tablename__ = "products"

    productCode = Column(String(15), primary_key=True)
    productName = Column(String(70), nullable=False)
    productLine = Column(String(50), ForeignKey("productlines.productLine"), nullable=False)
    productScale = Column(String(10), nullable=False)
    productVendor = Column(String(50), nullable=False)
    productDescription = Column(Text, nullable=False)
    quantityInStock = Column(Integer, nullable=False)
    buyPrice = Column(Numeric(10, 2), nullable=False)
    MSRP = Column(Numeric(10, 2), nullable=False)

    product_line_rel = relationship("ProductLine", back_populates="products")
    order_details = relationship("OrderDetail", back_populates="product")


class Office(Base):
    __tablename__ = "offices"

    officeCode = Column(String(10), primary_key=True)
    city = Column(String(50), nullable=False)
    phone = Column(String(50), nullable=False)
    addressLine1 = Column(String(50), nullable=False)
    addressLine2 = Column(String(50), nullable=True)
    state = Column(String(50), nullable=True)
    country = Column(String(50), nullable=False)
    postalCode = Column(String(15), nullable=False)
    territory = Column(String(10), nullable=False)

    employees = relationship("Employee", back_populates="office")


class Employee(Base):
    __tablename__ = "employees"

    employeeNumber = Column(Integer, primary_key=True)
    lastName = Column(String(50), nullable=False)
    firstName = Column(String(50), nullable=False)
    extension = Column(String(10), nullable=False)
    email = Column(String(100), nullable=False)
    officeCode = Column(String(10), ForeignKey("offices.officeCode"), nullable=False)
    reportsTo = Column(Integer, ForeignKey("employees.employeeNumber"), nullable=True)
    jobTitle = Column(String(50), nullable=False)

    office = relationship("Office", back_populates="employees")
    subordinates = relationship("Employee", backref="manager", remote_side=[employeeNumber])
    customers = relationship("Customer", back_populates="sales_rep")


class Customer(Base):
    __tablename__ = "customers"

    customerNumber = Column(Integer, primary_key=True)
    customerName = Column(String(50), nullable=False)
    contactLastName = Column(String(50), nullable=False)
    contactFirstName = Column(String(50), nullable=False)
    phone = Column(String(50), nullable=False)
    addressLine1 = Column(String(50), nullable=False)
    addressLine2 = Column(String(50), nullable=True)
    city = Column(String(50), nullable=False)
    state = Column(String(50), nullable=True)
    postalCode = Column(String(15), nullable=True)
    country = Column(String(50), nullable=False)
    salesRepEmployeeNumber = Column(
        Integer, ForeignKey("employees.employeeNumber"), nullable=True
    )
    creditLimit = Column(Numeric(10, 2), nullable=True)

    sales_rep = relationship("Employee", back_populates="customers")
    orders = relationship("Order", back_populates="customer")
    payments = relationship("Payment", back_populates="customer")


class Payment(Base):
    __tablename__ = "payments"

    customerNumber = Column(
        Integer, ForeignKey("customers.customerNumber"), primary_key=True
    )
    checkNumber = Column(String(50), primary_key=True)
    paymentDate = Column(Date, nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)

    customer = relationship("Customer", back_populates="payments")


class Order(Base):
    __tablename__ = "orders"

    orderNumber = Column(Integer, primary_key=True)
    orderDate = Column(Date, nullable=False)
    requiredDate = Column(Date, nullable=False)
    shippedDate = Column(Date, nullable=True)
    status = Column(String(15), nullable=False)
    comments = Column(Text, nullable=True)
    customerNumber = Column(
        Integer, ForeignKey("customers.customerNumber"), nullable=False
    )

    customer = relationship("Customer", back_populates="orders")
    order_details = relationship("OrderDetail", back_populates="order")


class OrderDetail(Base):
    __tablename__ = "orderdetails"

    orderNumber = Column(
        Integer, ForeignKey("orders.orderNumber"), primary_key=True
    )
    productCode = Column(
        String(15), ForeignKey("products.productCode"), primary_key=True
    )
    quantityOrdered = Column(Integer, nullable=False)
    priceEach = Column(Numeric(10, 2), nullable=False)
    orderLineNumber = Column(SmallInteger, nullable=False)

    order = relationship("Order", back_populates="order_details")
    product = relationship("Product", back_populates="order_details")
