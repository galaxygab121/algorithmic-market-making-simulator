"""marketmaker — a deterministic, offline, *simulated* market-making research environment.

Nothing in this package connects to a real exchange or broker. All prices and order
flow are synthetic (or replayed from a local CSV of mid prices) and every result is a
simulation output, not a statement about real-world profitability.
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "SIMULATED RESULTS — synthetic market data, simplified dynamics, paper trading only. "
    "Not investment advice and not indicative of real-world performance."
)
