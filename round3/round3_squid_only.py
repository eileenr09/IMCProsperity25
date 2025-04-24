from datamodel import OrderDepth, TradingState, Order
import jsonpickle
import statistics


import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()
outer_limit = 50
class Product:
    RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID = "SQUID_INK"

class Trader:
    def __init__(self):
        self.LIMIT = {
            Product.RESIN: 50,
            Product.KELP: 50,
            Product.SQUID: 50
        }

        self.price_history = {
            Product.SQUID: []
        }

        self.window_size = 5
        self.std_dev_multiplier = 0.85
        self.stop_loss_multiplier = 2.75
        self.max_trade_duration = 35

        self.active_trades = {
            Product.SQUID: None
        }

        self.tick_count = 0
        self.total_pnl = 0 

    def get_mid_price(self, order_depth):
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2

    def get_fair_value_and_bands(self, product, order_depth):
        current_price = self.get_mid_price(order_depth)
        if current_price is None:
            return None, None, None, None, None

        history = self.price_history[product]
        history.append(current_price)
        if len(history) > self.window_size:
            history.pop(0)

        sma = sum(history) / len(history)
        std_dev = statistics.stdev(history) if len(history) > 1 else 0

        upper_band = sma + self.std_dev_multiplier * std_dev
        lower_band = sma - self.std_dev_multiplier * std_dev

        return sma, upper_band, lower_band, current_price, std_dev

    def execute_trade(self, product, fair_value, upper_band, lower_band, current_price, std_dev, order_depth, state):
        orders = []

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(product, 0)
        limit = self.LIMIT[product]
        trade = self.active_trades[product]
        volatility_scale = max(0.1, 1 / (1 + std_dev))  # Prevents scaling too small
        adjusted_limit = int(limit * volatility_scale)
        adjusted_limit = limit

        # OPEN trade
        if trade is None:
            if current_price > upper_band:
                sell_qty = min(position + adjusted_limit, outer_limit)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    self.active_trades[product] = {
                        "position": "short",
                        "target_price": fair_value,
                        "entry_tick": self.tick_count,
                        "stop_loss": current_price + self.stop_loss_multiplier * std_dev
                    }

            elif current_price < lower_band:
                buy_qty = min(adjusted_limit - position, outer_limit)
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    self.active_trades[product] = {
                        "position": "long",
                        "target_price": fair_value,
                        "entry_tick": self.tick_count,
                        "stop_loss": current_price - self.stop_loss_multiplier * std_dev
                    }

        # CLOSE trade
        elif trade:
            time_in_trade = self.tick_count - trade["entry_tick"]
            stop_loss = trade["stop_loss"]
            exit_due_to_timeout = time_in_trade >= self.max_trade_duration

            if trade["position"] == "long":
                if current_price >= trade["target_price"] or current_price <= stop_loss or exit_due_to_timeout:
                    sell_qty = min(position, limit)
                    if sell_qty > 0:
                        orders.append(Order(product, best_bid, -sell_qty))
                        self.active_trades[product] = None

            elif trade["position"] == "short":
                if current_price <= trade["target_price"] or current_price >= stop_loss or exit_due_to_timeout:
                    buy_qty = min(limit + position, limit)
                    if buy_qty > 0:
                        orders.append(Order(product, best_ask, buy_qty))
                        self.active_trades[product] = None

        return orders

    def run(self, state):
        self.tick_count += 1
        result = {}

        product = Product.SQUID
        order_depth = state.order_depths.get(product)
        if order_depth is None:
            return {}, 0, ""

        fair_value, upper_band, lower_band, current_price, std_dev = self.get_fair_value_and_bands(product, order_depth)
        if fair_value is None:
            return {}, 0, ""

        orders = self.execute_trade(product, fair_value, upper_band, lower_band, current_price, std_dev, order_depth, state)
        result[product] = orders

        position = state.position.get(product, 0)
        pnl_last_tick = 0
        for order in orders:
            if order.quantity > 0:  # buy
                pnl_last_tick -= order.price * order.quantity
            else:  # sell
                pnl_last_tick -= order.price * order.quantity  # double negative

        self.total_pnl += pnl_last_tick
        logger.print(f"Tick: {self.tick_count} | Last PnL: {pnl_last_tick:.2f} | Total PnL: {self.total_pnl:.2f}")


        conversions = 0
        trader_data = jsonpickle.encode({
            "price_history": self.price_history,
            "active_trades": self.active_trades,
            "tick_count": self.tick_count
        })
        logger.flush(state, result, conversions, trader_data)
        return result, 0, trader_data