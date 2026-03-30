# env/data_generator.py
import random
from datetime import datetime, timedelta
from .models import Ticket, Priority, Department


TICKET_TEMPLATES = [
    # BILLING tickets
    {
        "subject": "Double charged on my invoice #{invoice}",
        "body": "Hello, I was charged twice for order #{order}. Amount: ${amount}. Please refund immediately. My account: {email}",
        "priority": Priority.HIGH,
        "department": Department.BILLING,
        "sentiment": "angry",
        "keywords": ["charged", "invoice", "refund", "amount"],
        "sla_hours": 4
    },
    {
        "subject": "Need copy of invoice for tax purposes",
        "body": "Hi, I need an invoice copy for my purchase made last month. Order ID: {order}. Thanks.",
        "priority": Priority.LOW,
        "department": Department.BILLING,
        "sentiment": "neutral",
        "keywords": ["invoice", "copy", "tax"],
        "sla_hours": 48
    },
    # TECHNICAL tickets
    {
        "subject": "URGENT: Production server is DOWN",
        "body": "Our entire production environment is down! Error: 500 Internal Server Error. This is affecting all our customers. Need immediate help!",
        "priority": Priority.CRITICAL,
        "department": Department.TECHNICAL,
        "sentiment": "angry",
        "keywords": ["down", "urgent", "production", "server", "error"],
        "sla_hours": 1
    },
    {
        "subject": "How to reset my password?",
        "body": "Hi, I forgot my password and can't login. Can you help me reset it? Thanks!",
        "priority": Priority.LOW,
        "department": Department.TECHNICAL,
        "sentiment": "neutral",
        "keywords": ["password", "reset", "login"],
        "sla_hours": 24
    },
    {
        "subject": "App crashes when uploading files over 10MB",
        "body": "Getting an error when I try to upload files larger than 10MB. Error message: 'Upload failed'. This is blocking my work.",
        "priority": Priority.MEDIUM,
        "department": Department.TECHNICAL,
        "sentiment": "neutral",
        "keywords": ["crash", "upload", "error", "file"],
        "sla_hours": 8
    },
    # RETURNS tickets
    {
        "subject": "Return request - wrong item received",
        "body": "I ordered product A but received product B. Order #{order}. I need to return this and get the correct item ASAP.",
        "priority": Priority.HIGH,
        "department": Department.RETURNS,
        "sentiment": "angry",
        "keywords": ["return", "wrong", "order", "received"],
        "sla_hours": 6
    },
    {
        "subject": "Damaged product on delivery",
        "body": "My package arrived damaged. The product inside is broken. I'd like a replacement or refund. Order: {order}",
        "priority": Priority.HIGH,
        "department": Department.RETURNS,
        "sentiment": "angry",
        "keywords": ["damaged", "broken", "replacement", "refund"],
        "sla_hours": 6
    },
    # GENERAL tickets
    {
        "subject": "How do I export my data?",
        "body": "Hi, I'd like to know how to export all my data from your platform. Is there a CSV export option?",
        "priority": Priority.LOW,
        "department": Department.GENERAL,
        "sentiment": "neutral",
        "keywords": ["export", "data", "csv"],
        "sla_hours": 48
    },
]


def generate_ticket(ticket_id: str, template_idx: int = None) -> Ticket:
    if template_idx is None:
        template_idx = random.randint(0, len(TICKET_TEMPLATES) - 1)
    
    template = TICKET_TEMPLATES[template_idx % len(TICKET_TEMPLATES)]
    
    # Fill placeholders
    order_id = f"ORD-{random.randint(10000, 99999)}"
    invoice_id = f"INV-{random.randint(1000, 9999)}"
    amount = random.randint(50, 500)
    email = f"customer{random.randint(100, 999)}@example.com"
    
    subject = template["subject"].format(
        invoice=invoice_id, order=order_id, amount=amount, email=email
    )
    body = template["body"].format(
        invoice=invoice_id, order=order_id, amount=amount, email=email
    )
    
    return Ticket(
        ticket_id=ticket_id,
        subject=subject,
        body=body,
        customer_name=f"Customer {random.randint(100, 999)}",
        customer_email=email,
        created_at=datetime.now().isoformat(),
        sla_deadline_hours=template["sla_hours"],
        true_priority=template["priority"],
        true_department=template["department"],
        keywords=template["keywords"],
        sentiment=template["sentiment"]
    )


def generate_ticket_batch(count: int, seed: int = 42) -> list:
    random.seed(seed)
    tickets = []
    for i in range(count):
        ticket = generate_ticket(
            ticket_id=f"TKT-{1000 + i}",
            template_idx=i
        )
        tickets.append(ticket)
    return tickets
