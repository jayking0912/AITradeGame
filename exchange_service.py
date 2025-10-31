"""
Simple exchange client abstractions built on ccxt for Binance and OKX.

These helpers provide synchronous wrappers to query key account data such as
balances, trading fees, and leverage limits that the trading engine needs in
order to mirror real exchange constraints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import ccxt


logger = logging.getLogger(__name__)


class ExchangeClientError(Exception):
    """Wrapper for exchange client specific errors."""


@dataclass
class ExchangeCredentials:
    """Container for exchange connection details."""

    name: str
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    use_sandbox: bool = False
    account_type: Optional[str] = None  # e.g. 'spot', 'futures', 'swap'
    quote_currency: str = 'USDT'


class ExchangeClient:
    """
    Lightweight synchronous wrapper around ccxt exchanges.

    Responsibilities:
    - establish exchange connection
    - expose helpers for fee rate, leverage, and balances
    """

    def __init__(self, credentials: ExchangeCredentials):
        self.credentials = credentials
        self.exchange = self._create_exchange()
        self._ensure_markets_loaded()

    # ------------------------------------------------------------------ #
    # Exchange bootstrap helpers
    # ------------------------------------------------------------------ #

    def _create_exchange(self):
        name = self.credentials.name.lower()
        api_config = {
            'apiKey': self.credentials.api_key,
            'secret': self.credentials.api_secret,
        }

        if self.credentials.passphrase:
            api_config['password'] = self.credentials.passphrase

        account_type = (self.credentials.account_type or 'spot').lower()

        # Select appropriate ccxt exchange class
        if name == 'binance' and account_type in {'futures', 'swap', 'usdm'}:
            exchange = ccxt.binanceusdm(api_config)
        elif name == 'binance':
            exchange = ccxt.binance(api_config)
        elif name in {'binanceusdm', 'binance-futures', 'binance_future'}:
            exchange = ccxt.binanceusdm(api_config)
        elif name == 'okx':
            exchange = ccxt.okx(api_config)
        else:
            raise ExchangeClientError(f"Unsupported exchange: {self.credentials.name}")

        if hasattr(exchange, 'options'):
            if name.startswith('binance'):
                exchange.options.setdefault('defaultType', account_type)
            if name == 'okx':
                exchange.options.setdefault('defaultType', 'spot' if account_type == 'spot' else 'swap')

        try:
            exchange.enableRateLimit = True
            exchange.timeout = 60_000
            if self.credentials.use_sandbox and hasattr(exchange, 'set_sandbox_mode'):
                exchange.set_sandbox_mode(True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to configure exchange sandbox/rate-limit: %s", exc)

        return exchange

    def _ensure_markets_loaded(self):
        """Load market metadata once to access leverage limits, etc."""
        try:
            self.exchange.load_markets()
        except Exception as exc:
            raise ExchangeClientError(f"Failed to load markets: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    def get_balance_snapshot(self) -> Dict[str, float]:
        """Return free/used/total balances for the configured quote currency."""
        quote = self.credentials.quote_currency.upper()
        try:
            balance = self.exchange.fetch_balance()
        except Exception as exc:
            raise ExchangeClientError(f"Failed to fetch balance: {exc}") from exc

        result = {
            'free': float(balance.get('free', {}).get(quote, 0.0)),
            'used': float(balance.get('used', {}).get(quote, 0.0)),
            'total': float(balance.get('total', {}).get(quote, 0.0)),
        }

        # Some exchanges report totals only via info field; fallback if necessary
        if result['total'] == 0 and result['free'] == 0 and result['used'] == 0:
            info = balance.get(quote) or balance.get(quote.lower()) or {}
            if isinstance(info, dict):
                for key in ('total', 'free', 'used'):
                    if key in info:
                        result[key] = float(info[key])

        result['available'] = result['free']
        return result

    def get_taker_fee(self, symbol: str) -> float:
        """Get taker fee for the given symbol, fallback to exchange default."""
        market_symbol = self._resolve_symbol(symbol)
        try:
            fee_info = self.exchange.fetch_trading_fee(market_symbol)
            taker = fee_info.get('taker')
            if taker is not None:
                return float(taker)
        except Exception as exc:
            logger.debug("fetch_trading_fee failed for %s: %s", market_symbol, exc)

        # Fallback to global taker fee if available
        taker_fee = (
            self.exchange.fees.get('trading', {}).get('taker')
            if hasattr(self.exchange, 'fees') else None
        )
        if taker_fee is not None:
            return float(taker_fee)

        # Default to 0.1% if nothing available
        return 0.001

    def get_max_leverage(self, symbol: str) -> float:
        """Infer maximum leverage allowed for the given symbol."""
        market_symbol = self._resolve_symbol(symbol)
        market = self.exchange.market(market_symbol)

        leverage = None
        limits = market.get('limits', {})
        if isinstance(limits, dict):
            leverage_data = limits.get('leverage')
            if isinstance(leverage_data, dict):
                leverage = leverage_data.get('max')

        if leverage is None:
            leverage = market.get('maxLeverage') or market.get('leverage')

        if leverage is None:
            info = market.get('info', {})
            if isinstance(info, dict):
                leverage = (
                    info.get('maxLeverage')
                    or info.get('max_leverage')
                    or info.get('leverage')
                )

        try:
            leverage_value = float(leverage) if leverage is not None else None
        except (TypeError, ValueError):
            leverage_value = None

        if leverage_value and leverage_value > 0:
            return leverage_value

        # Default leverage when exchange does not provide explicit values.
        return 1.0 if (self.credentials.account_type or 'spot').lower() == 'spot' else 10.0

    def _resolve_symbol(self, symbol: str) -> str:
        """Resolve REST symbol, defaulting to quote currency suffix."""
        if '/' in symbol:
            return symbol
        quote = self.credentials.quote_currency.upper()
        candidate = f"{symbol}/{quote}"
        if candidate in self.exchange.markets:
            return candidate
        candidate = f"{symbol}-{quote}"
        if candidate in self.exchange.markets:
            return candidate
        return symbol

    def close(self):
        """Close exchange client if supported."""
        try:
            if hasattr(self.exchange, 'close'):
                self.exchange.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to close exchange client: %s", exc)


def create_exchange_client(credentials: ExchangeCredentials) -> ExchangeClient:
    """Factory helper to create exchange client with validation."""
    if not credentials.api_key or not credentials.api_secret:
        raise ExchangeClientError("API key/secret are required for exchange trading")
    return ExchangeClient(credentials)
