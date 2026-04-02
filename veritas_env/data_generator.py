# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Synthetic financial crime scenario generator for Veritas AI.

Every call to generate_scenario() produces a fresh, self-contained
investigation case. The output is a plain dict that environment.py
stores and queries — no external database required.

Design rules:
  - Deterministic: same seed always returns identical scenario
  - Self-contained: all data lives in the returned dict
  - Realistic: amounts, dates, merchants match real-world patterns
  - Noisy: innocent accounts always present as distraction
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List


# ─────────────────────────────────────────────────────────
# REALISTIC DATA POOLS
# ─────────────────────────────────────────────────────────

FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Neha",
    "Arjun", "Kavya", "Siddharth", "Meera", "Raj", "Divya",
    "Kiran", "Sunita", "Amit", "Pooja", "Dev", "Riya",
    "Harsh", "Nisha", "Varun", "Sneha", "Anil", "Preeti",
]

LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Gupta", "Joshi",
    "Mehta", "Shah", "Verma", "Nair", "Reddy", "Rao",
    "Iyer", "Bhat", "Malhotra", "Chopra", "Das", "Sen",
]

MERCHANTS_NORMAL = [
    "BigMart Grocery", "FuelStop", "MediPlus Pharmacy",
    "BookWorld", "CafeBean", "ElectroMart",
    "FreshVeggies", "QuickBite Restaurant", "CityBus Pass",
    "StreamFlix Subscription", "TelecomRecharge", "GymFit Monthly",
]

MERCHANTS_SUSPICIOUS = [
    "LuxeJewels International", "CryptoXchange Portal",
    "HighStakes Gaming", "QuickCash ATM Network",
    "Anonymous Transfer Co", "OffshoreGift Cards",
    "RapidPawn Solutions", "NightMarket Vendor",
]

REGIONS = ["Mumbai", "Delhi", "Bangalore", "Chennai",
           "Hyderabad", "Pune", "Kolkata", "Ahmedabad"]

DEVICE_PREFIXES = ["DEV", "MOB", "TAB", "WEB"]
IP_PREFIXES = ["192.168", "10.0", "172.16", "203.45"]


# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────

def _make_account_id(rng: random.Random) -> str:
    """Generate a realistic account ID like ACC-4821."""
    return f"ACC-{rng.randint(1000, 9999)}"


def _make_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _make_device_id(rng: random.Random) -> str:
    prefix = rng.choice(DEVICE_PREFIXES)
    return f"{prefix}-{rng.randint(10000, 99999)}"


def _make_ip(rng: random.Random) -> str:
    prefix = rng.choice(IP_PREFIXES)
    return f"{prefix}.{rng.randint(1,254)}.{rng.randint(1,254)}"


def _random_date(rng: random.Random,
                 start: datetime,
                 end: datetime) -> str:
    delta = end - start
    offset = rng.randint(0, delta.days)
    return (start + timedelta(days=offset)).strftime("%Y-%m-%d")


def _make_account(
    rng: random.Random,
    account_id: str,
    device_id: str = None,
    ip: str = None,
) -> Dict[str, Any]:
    """Create one account profile."""
    return {
        "account_id":        account_id,
        "name":              _make_name(rng),
        "email":             f"user{rng.randint(100,999)}@email.com",
        "phone":             f"+91-{rng.randint(7000000000,9999999999)}",
        "region":            rng.choice(REGIONS),
        "account_type":      rng.choice(["savings", "current", "wallet"]),
        "balance":           round(rng.uniform(500, 50000), 2),
        "registration_date": _random_date(
            rng,
            datetime(2020, 1, 1),
            datetime(2023, 12, 31)
        ),
        "device_id":  device_id or _make_device_id(rng),
        "ip_address": ip or _make_ip(rng),
        "status":     "active",
    }


def _make_transaction(
    rng: random.Random,
    txn_id: str,
    from_account: str,
    to_account: str,
    amount: float,
    date: str,
    merchant: str,
    is_suspicious: bool = False,
) -> Dict[str, Any]:
    """Create one transaction record."""
    return {
        "txn_id":        txn_id,
        "from_account":  from_account,
        "to_account":    to_account,
        "amount":        round(amount, 2),
        "date":          date,
        "merchant":      merchant,
        "region":        rng.choice(REGIONS),
        "status":        "completed",
        "risk_score":    round(rng.uniform(0.7, 1.0), 2)
                         if is_suspicious
                         else round(rng.uniform(0.0, 0.3), 2),
    }


def _make_alert(
    alert_id: str,
    account_id: str,
    alert_type: str,
    severity: str,
    description: str,
) -> Dict[str, Any]:
    return {
        "alert_id":    alert_id,
        "account_id":  account_id,
        "alert_type":  alert_type,
        "severity":    severity,
        "description": description,
    }


# ─────────────────────────────────────────────────────────
# SCENARIO BUILDERS
# ─────────────────────────────────────────────────────────

def _build_card_scheme(rng: random.Random) -> Dict[str, Any]:
    """
    Task 1 — Easy.
    One account makes rapid high-value purchases at suspicious
    merchants. Clear velocity anomaly, easy to spot.
    """
    start = datetime(2024, 1, 1)
    end   = datetime(2024, 3, 31)

    # Suspect account
    suspect_id = _make_account_id(rng)
    suspect     = _make_account(rng, suspect_id)

    # 4–6 innocent accounts as noise
    n_innocent = rng.randint(4, 6)
    innocent_ids = [_make_account_id(rng) for _ in range(n_innocent)]
    # Ensure no duplicates
    innocent_ids = list(set(innocent_ids) - {suspect_id})
    innocents = [_make_account(rng, aid) for aid in innocent_ids]

    all_accounts = {acc["account_id"]: acc
                    for acc in [suspect] + innocents}

    # Suspicious transactions — rapid purchases at dodgy merchants
    transactions = []
    alerts = []
    n_suspicious = rng.randint(6, 10)
    for i in range(n_suspicious):
        txn_date = _random_date(rng, start, start + timedelta(days=14))
        transactions.append(_make_transaction(
            rng,
            txn_id       = f"TXN-{rng.randint(10000,99999)}",
            from_account = suspect_id,
            to_account   = "MERCHANT",
            amount       = rng.uniform(8000, 15000),
            date         = txn_date,
            merchant     = rng.choice(MERCHANTS_SUSPICIOUS),
            is_suspicious= True,
        ))

    # Normal transactions for innocent accounts
    for acc in innocents:
        for _ in range(rng.randint(2, 4)):
            transactions.append(_make_transaction(
                rng,
                txn_id       = f"TXN-{rng.randint(10000,99999)}",
                from_account = acc["account_id"],
                to_account   = "MERCHANT",
                amount       = rng.uniform(100, 2000),
                date         = _random_date(rng, start, end),
                merchant     = rng.choice(MERCHANTS_NORMAL),
                is_suspicious= False,
            ))

    alerts.append(_make_alert(
        alert_id    = "ALT-001",
        account_id  = suspect_id,
        alert_type  = "velocity_anomaly",
        severity    = "high",
        description = (
            f"Account {suspect_id} triggered 6+ high-value "
            "transactions within 14 days at unlisted merchants."
        ),
    ))

    return {
        "case_type":          "card_scheme",
        "primary_suspect":    suspect_id,
        "associates":         [],
        "accounts":           all_accounts,
        "transactions":       transactions,
        "alerts":             alerts,
        "accounts_in_scope":  list(all_accounts.keys()),
    }


def _build_layering_scheme(rng: random.Random) -> Dict[str, Any]:
    """
    Task 2 — Medium.
    Money flows through a chain of 3–4 accounts. Each transfer
    is just under the reporting threshold (₹10,000).
    Structuring pattern — requires following the money.
    """
    start = datetime(2024, 1, 1)
    end   = datetime(2024, 3, 31)

    # Build the chain: origin → layer1 → layer2 → (layer3) → destination
    chain_len   = rng.randint(3, 4)
    chain_ids   = [_make_account_id(rng) for _ in range(chain_len)]
    chain_ids   = list(dict.fromkeys(chain_ids))   # deduplicate, keep order
    suspect_id  = chain_ids[0]
    associates  = chain_ids[1:]

    chain_accounts = {cid: _make_account(rng, cid) for cid in chain_ids}

    # 4–5 innocent accounts
    innocent_ids = [_make_account_id(rng) for _ in range(rng.randint(4,5))]
    innocent_ids = list(set(innocent_ids) - set(chain_ids))
    innocents    = [_make_account(rng, aid) for aid in innocent_ids]

    all_accounts = {**chain_accounts,
                    **{acc["account_id"]: acc for acc in innocents}}

    # Chain transactions — structuring just under ₹10,000
    transactions = []
    current_date = start
    for i in range(len(chain_ids) - 1):
        n_txns = rng.randint(3, 5)
        for _ in range(n_txns):
            transactions.append(_make_transaction(
                rng,
                txn_id       = f"TXN-{rng.randint(10000,99999)}",
                from_account = chain_ids[i],
                to_account   = chain_ids[i + 1],
                amount       = rng.uniform(8500, 9800),   # just under ₹10k
                date         = _random_date(
                    rng, current_date,
                    current_date + timedelta(days=20)
                ),
                merchant     = "Peer Transfer",
                is_suspicious= True,
            ))
        current_date += timedelta(days=15)

    # Normal transactions for innocent accounts
    for acc in innocents:
        for _ in range(rng.randint(2, 3)):
            transactions.append(_make_transaction(
                rng,
                txn_id       = f"TXN-{rng.randint(10000,99999)}",
                from_account = acc["account_id"],
                to_account   = "MERCHANT",
                amount       = rng.uniform(200, 3000),
                date         = _random_date(rng, start, end),
                merchant     = rng.choice(MERCHANTS_NORMAL),
                is_suspicious= False,
            ))

    alerts = [_make_alert(
        alert_id    = "ALT-002",
        account_id  = suspect_id,
        alert_type  = "structuring_pattern",
        severity    = "medium",
        description = (
            f"Account {suspect_id} shows repeated peer transfers "
            "just below the ₹10,000 reporting threshold."
        ),
    )]

    return {
        "case_type":         "layering_scheme",
        "primary_suspect":   suspect_id,
        "associates":        associates,
        "accounts":          all_accounts,
        "transactions":      transactions,
        "alerts":            alerts,
        "accounts_in_scope": list(all_accounts.keys()),
    }


def _build_coordinated_scheme(rng: random.Random) -> Dict[str, Any]:
    """
    Task 3 — Hard.
    5–8 accounts controlled by the same operator, linked by
    shared device IDs and IP address ranges. Requires
    cross-referencing device and IP data to find the ring.
    """
    start = datetime(2024, 1, 1)
    end   = datetime(2024, 3, 31)

    ring_size   = rng.randint(5, 8)
    ring_ids    = [_make_account_id(rng) for _ in range(ring_size)]
    ring_ids    = list(dict.fromkeys(ring_ids))
    suspect_id  = ring_ids[0]
    associates  = ring_ids[1:]

    # All ring accounts share device IDs and IP subnet
    shared_device = _make_device_id(rng)
    shared_ip_prefix = rng.choice(IP_PREFIXES)

    ring_accounts = {}
    for i, rid in enumerate(ring_ids):
        # Alternate between shared device and unique device
        device = shared_device if i % 2 == 0 else _make_device_id(rng)
        ip = (f"{shared_ip_prefix}."
              f"{rng.randint(1,10)}."
              f"{rng.randint(1,254)}")
        ring_accounts[rid] = _make_account(rng, rid, device, ip)

    # 5–6 innocent accounts with distinct devices and IPs
    innocent_ids = [_make_account_id(rng) for _ in range(rng.randint(5,6))]
    innocent_ids = list(set(innocent_ids) - set(ring_ids))
    innocents    = [_make_account(rng, aid) for aid in innocent_ids]

    all_accounts = {**ring_accounts,
                    **{acc["account_id"]: acc for acc in innocents}}

    # Ring transactions — coordinated transfers to external accounts
    transactions = []
    external = f"EXT-{rng.randint(1000,9999)}"
    for rid in ring_ids:
        for _ in range(rng.randint(2, 4)):
            transactions.append(_make_transaction(
                rng,
                txn_id       = f"TXN-{rng.randint(10000,99999)}",
                from_account = rid,
                to_account   = external,
                amount       = rng.uniform(5000, 25000),
                date         = _random_date(rng, start, end),
                merchant     = rng.choice(MERCHANTS_SUSPICIOUS),
                is_suspicious= True,
            ))

    # Normal transactions for innocents
    for acc in innocents:
        for _ in range(rng.randint(2, 3)):
            transactions.append(_make_transaction(
                rng,
                txn_id       = f"TXN-{rng.randint(10000,99999)}",
                from_account = acc["account_id"],
                to_account   = "MERCHANT",
                amount       = rng.uniform(100, 5000),
                date         = _random_date(rng, start, end),
                merchant     = rng.choice(MERCHANTS_NORMAL),
                is_suspicious= False,
            ))

    alerts = [_make_alert(
        alert_id    = "ALT-003",
        account_id  = suspect_id,
        alert_type  = "coordinated_activity",
        severity    = "critical",
        description = (
            f"Multiple accounts including {suspect_id} show "
            "coordinated transfer patterns to a single external account."
        ),
    )]

    return {
        "case_type":         "coordinated_scheme",
        "primary_suspect":   suspect_id,
        "associates":        associates,
        "accounts":          all_accounts,
        "transactions":      transactions,
        "alerts":            alerts,
        "accounts_in_scope": list(all_accounts.keys()),
    }


# ─────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# Called by environment.py on every reset()
# ─────────────────────────────────────────────────────────

TASK_TO_SCHEME = {
    "task_easy":   _build_card_scheme,
    "task_medium": _build_layering_scheme,
    "task_hard":   _build_coordinated_scheme,
}


def generate_scenario(task_id: str, seed: int) -> Dict[str, Any]:
    """
    Generate a fresh investigation scenario.

    Args:
        task_id : one of task_easy | task_medium | task_hard
        seed    : random seed — same seed always gives same scenario

    Returns:
        dict with keys:
          case_type, primary_suspect, associates,
          accounts, transactions, alerts, accounts_in_scope
    """
    rng = random.Random(seed)

    builder = TASK_TO_SCHEME.get(task_id)
    if builder is None:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Must be one of: {list(TASK_TO_SCHEME.keys())}"
        )

    scenario = builder(rng)
    scenario["case_id"] = f"CASE-{seed}-{task_id}"
    scenario["seed"]    = seed

    return scenario