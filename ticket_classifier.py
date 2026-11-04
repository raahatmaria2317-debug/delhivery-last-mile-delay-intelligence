
DEPARTMENTS = {
    "Delivery": "Delivery department",
    "Payment": "Payment department",
    "Needs Review": "Support executive",
}

TICKETS = [
    "The laptop has not arrived yet",
    "Please process my refund for the mouse",
    "The keyboard keys feel loose",
    "Payment done but delivery is late",
    "REFUND NOT RECEIVED",
]


def summarize_ticket(ticket):
    return "Ticket received: " + ticket


def classify_ticket(ticket):
    ticket = ticket.lower()

    if "delivery" in ticket or "arrived" in ticket:
        return "Delivery"
    elif "payment" in ticket or "refund" in ticket:
        return "Payment"
    else:
        return "Needs Review"


def create_ticket_report(ticket):
    summary = summarize_ticket(ticket)
    category = classify_ticket(ticket)
    department = DEPARTMENTS[category]

    return {
        "summary": summary,
        "category": category,
        "department": department
    }


for ticket in TICKETS:
    print(create_ticket_report(ticket))

DEPARTMENTS = {
    "Delivery": "Delivery department",
    "Payment": "Payment department",
    "Needs Review": "Support executive",
}

TICKETS = [
    "The laptop has not arrived yet",
    "Please process my refund for the mouse",
    "The keyboard keys feel loose",
    "Payment done but delivery is late",
    "REFUND NOT RECEIVED",
]


# Step 1: Summarize the ticket
def summarize_ticket(ticket):
    return "Ticket received: " + ticket


# Step 2: Classify the ticket
def classify_ticket(ticket):
    ticket = ticket.lower()

    # Check Delivery first
    if "delivery" in ticket or "arrived" in ticket:
        return "Delivery"

    # Check Payment if Delivery does not match
    elif "payment" in ticket or "refund" in ticket:
        return "Payment"

    # If neither matches
    else:
        return "Needs Review"


# Step 3: Create the final ticket report
def create_ticket_report(ticket):
    summary = summarize_ticket(ticket)
    category = classify_ticket(ticket)
    department = DEPARTMENTS[category]

    return {
        "summary": summary,
        "category": category,
        "department": department
    }


# Step 4: Run the function chain
for ticket in TICKETS:
    print(create_ticket_report(ticket))


# Bonus question
bonus_ticket = "My parcel never came"
print(create_ticket_report(bonus_ticket))
