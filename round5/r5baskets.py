
from typing import Dict
from collections import deque
from enum import IntEnum
from abc import abstractmethod
from datamodel import Order, OrderDepth, TradingState

JSON = dict
Symbol = str


class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    def act(self, state: TradingState) -> None:
        raise NotImplementedError("You must implement act() method in subclasses.")

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.orders = []
        self.conversions = 0

        self.act(state)

        return self.orders, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.symbol, price, -quantity))

    def convert(self, amount: int) -> None:
        self.conversions += amount

    def get_mid_price(self, state: TradingState, symbol: str) -> float:
        order_depth = state.order_depths[symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return (popular_buy_price + popular_sell_price) / 2

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass


class Signal:
    NEUTRAL = 0
    SHORT = 1
    LONG = 2

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, Signal):
            return self.value == other.value
        return self.value == other

    def __repr__(self):
        return f"Signal({self.value})"


class SignalStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.signal = Signal.NEUTRAL

    def get_signal(self, state: TradingState) -> Signal | None:
        raise NotImplementedError("You must implement get_signal() in SignalStrategy subclasses.")
    def act(self, state: TradingState) -> None:
        new_signal = self.get_signal(state)
        if new_signal is not None:
            self.signal = new_signal

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        if self.signal == Signal.NEUTRAL:
            if position < 0:
                self.buy(self.get_buy_price(order_depth), -position)
            elif position > 0:
                self.sell(self.get_sell_price(order_depth), position)
        elif self.signal == Signal.SHORT:
            self.sell(self.get_sell_price(order_depth), self.limit + position)
        elif self.signal == Signal.LONG:
            self.buy(self.get_buy_price(order_depth), self.limit - position)

    def get_buy_price(self, order_depth: OrderDepth) -> int:
        return min(order_depth.sell_orders.keys())

    def get_sell_price(self, order_depth: OrderDepth) -> int:
        return max(order_depth.buy_orders.keys())

    def save(self) -> JSON:
        return self.signal

    def load(self, data: JSON) -> None:
        self.signal = Signal(data)



class Basket1Strategy(SignalStrategy):
    def get_signal(self, state: TradingState) -> Signal | None:
        if any(symbol not in state.order_depths for symbol in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1"]):
            return

        croissants = self.get_mid_price(state, "CROISSANTS")
        jams = self.get_mid_price(state, "JAMS")
        djembes = self.get_mid_price(state, "DJEMBES")
        basket1 = self.get_mid_price(state, "PICNIC_BASKET1")

        diff = basket1 - 6 * croissants - 3 * jams - djembes

        if diff < -12:
            return Signal.LONG
        elif diff > 51:
            return Signal.SHORT


class Basket2Strategy(SignalStrategy):
    def get_signal(self, state: TradingState) -> Signal | None:
        if any(symbol not in state.order_depths for symbol in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET2"]):
            return

        croissants = self.get_mid_price(state, "CROISSANTS")
        jams = self.get_mid_price(state, "JAMS")
        djembes = self.get_mid_price(state, "DJEMBES")
        basket2 = self.get_mid_price(state, "PICNIC_BASKET2")

        diff = basket2 - 4 * croissants - 2 * jams

        if diff < 61:
            return Signal.LONG
        elif diff > 148:
            return Signal.SHORT


# class Basket1Strategy(SignalStrategy):
#     def __init__(self, symbol: str, limit: int) -> None:
#         super().__init__(symbol, limit)
#         self.signal_map = {
#             ("Pablo", "Caesar"): Signal.LONG,
#             ("Camilla", "Pablo"): Signal.LONG,
#             ("Caesar", "Penelope"): Signal.LONG,

#             ("Caesar", "Pablo"): Signal.SHORT,
#             ("Camilla", "Penelope"): Signal.SHORT,
#         }

#     def get_signal(self, state: TradingState) -> Signal | None:
#         trades = state.market_trades.get(self.symbol, [])
#         trades = [t for t in trades if t.timestamp == state.timestamp - 200]
#         for trade in reversed(trades):
#             pair = (trade.seller, trade.buyer)
#             if pair in self.signal_map:
#                 return self.signal_map[pair]
#         return None


# class Basket2Strategy(SignalStrategy):
#     def __init__(self, symbol: str, limit: int) -> None:
#         super().__init__(symbol, limit)
#         self.signal_map = {
#             ("Penelope", "Camilla"): Signal.LONG,

#             ("Pablo", "Caesar"): Signal.SHORT,
#             ("Camilla", "Pablo"): Signal.SHORT,
#             ("Camilla", "Penelope"): Signal.SHORT,
#         }

#     def get_signal(self, state: TradingState) -> Signal | None:
#         trades = state.market_trades.get(self.symbol, [])
#         trades = [t for t in trades if t.timestamp == state.timestamp - 200]
#         for trade in reversed(trades):
#             pair = (trade.seller, trade.buyer)
#             if pair in self.signal_map:
#                 return self.signal_map[pair]
#         return None

class Trader:
    def __init__(self) -> None:
        limits = {
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
        }

        self.strategies: Dict[Symbol, Strategy] = {
            "PICNIC_BASKET1": Basket1Strategy("PICNIC_BASKET1", limits["PICNIC_BASKET1"]),
            "PICNIC_BASKET2": Basket2Strategy("PICNIC_BASKET2", limits["PICNIC_BASKET2"]),
        }

    def run(self, state: TradingState) -> tuple[Dict[Symbol, list[Order]], int, str]:
        orders = {}
        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if symbol in state.order_depths and state.order_depths[symbol].buy_orders and state.order_depths[symbol].sell_orders:
                strategy_orders, strategy_conversions = strategy.run(state)
                orders[symbol] = strategy_orders
                conversions += strategy_conversions

            new_trader_data[symbol] = strategy.save()


        return orders, conversions, ""