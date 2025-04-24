from typing import Dict, List
from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import jsonpickle
import numpy as np
import math
from statistics import NormalDist
import statistics

JSON = dict
Symbol = str

class Product:
    MACARONS = "MAGNIFICENT_MACARONS"
    SQUID = "SQUID_INK"

outer_limit = 50
LIMIT_BSK1 = 60
LIMIT_BSK2 = 100

PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"

RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
CROISSANTS = "CROISSANTS"
JAMS = "JAMS"
DJEMBE = "DJEMBE"
MACARON = "MAGNIFICENT_MACARONS"

# Option
strikes = [9500, 9750, 10000, 10250, 10500]

ROCK = "VOLCANIC_ROCK"
ROCK_VOUCHERS = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in strikes]
    
PARAMS = {
    KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    "VOLCANIC_ROCK_VOUCHER_9500": {
        "mean_volatility": 0.03035871041620843,
        "strike": 9500,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 30,
        "zscore_threshold": 2.1
    },
    "VOLCANIC_ROCK_VOUCHER_9750": {
        "mean_volatility": 0.025696009043535226,
        "strike": 9750,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 30,
        "zscore_threshold": 2.1
    },
    "VOLCANIC_ROCK_VOUCHER_10000": {
        "mean_volatility": 0.025274225980310268,
        "strike": 10000,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 30,
        "zscore_threshold": 2.1
    },
    "VOLCANIC_ROCK_VOUCHER_10250": {
        "mean_volatility": 0.027508611365627284,
        "strike": 10250,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 30,
        "zscore_threshold": 2.1
    },
    "VOLCANIC_ROCK_VOUCHER_10500": {
        "mean_volatility": 0.038103486570811695,
        "strike": 10500,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 30,
        "zscore_threshold": 2.1
    },
    "VOLCANIC_ROCK": {
        "mean_volatility": 0.02711458250986778,
        "strike": 10500,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 30,
        "zscore_threshold": 2.1
    },

    Product.MACARONS:{
        "make_edge": 2,
        "make_min_edge": 1,
        "make_probability": 0.566,
        "init_make_edge": 2,
        "min_edge": 0.5,
        "volume_avg_timestamp": 5,
        "volume_bar": 75,
        "dec_edge_discount": 0.8,
        "step_size":0.5
    }
}


class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    def act(self, state: TradingState) -> None:
        raise NotImplementedError("You must implement act() method in subclasses.")

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        self.result = []
        self.conversions = 0

        self.act(state)

        return self.result, self.conversions

    def buy(self, price: int, quantity: int) -> None:
        self.result.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.result.append(Order(self.symbol, price, -quantity))

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

from math import log, sqrt, exp
from statistics import NormalDist

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))

        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


THRESHOLD_HIGH = 0.8
THRESHOLD_LOW = -0.8
PRICE_OFFSET = 1  # Shift bid/ask slightly to increase chance of execution
ALPHA = 0.1
VOL_WINDOW = 20

LIMITS = {RESIN: 50, KELP: 50, Product.SQUID: 50, PICNIC_BASKET1: LIMIT_BSK1, PICNIC_BASKET2: LIMIT_BSK2}


K = {
    RESIN: 1,
    Product.SQUID: 1,
    PICNIC_BASKET1: 1,
    PICNIC_BASKET2: 1,
    KELP: 0.5,
}

MIN_VOLUME = 5  # Minimum volume to trade
BASE_SPREAD = 1.0  # Base price offset to increase fill chance
SPREAD_EDGE = 1  # Offset to increase fill chance

class Trader:
    def __init__(self, params=None):
        self.ema = {}
        self.ma = {}
        self.volatility = {}
        self.price_history_volatility = {}
        self.price_history_ma = {}
        if params is None:
            params = PARAMS
        self.params = params
        self.recipes = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2},
        }

        self.limits = {RESIN: 50,
            KELP: 50,
            Product.SQUID: 50,
            "CROISSANTS": 250,
            "JAMS":   350,
            "DJEMBES":    60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2":100,
            ROCK: 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            Product.MACARONS: 10
        }
        self.price_history = {
            Product.SQUID: [],
            PICNIC_BASKET1: [],
            PICNIC_BASKET2: []
        }
        self.strikes = strikes
        self.voucher_products = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in self.strikes]
        self.rock_product = "VOLCANIC_ROCK"


        # SQUID PARAMS
        # EWMA parameters for mid‑price and variance
        self.alpha_squid = 0.2
        # Market‑making parameters
        self.spread_multiplier_squid = 2.0   # half‑spread = multiplier × σ
        self.min_spread_squid        = 1     # minimum half‑spread
        self.max_spread_squid      = 10    # maximum half‑spread
        self.base_order_size_squid   = 20    # nominal size per quote
        self.min_order_size_squid    = 5
        self.max_order_size_squid    = 50
        self.inventory_risk_squid    = 0.1   # ticks skew per unit

        # Persistent EWMA state
        self.ewma_mid_squid = None
        self.ewma_var_squid = None

        self.tick_count = 0

        # ETF
        self.strategies: Dict[Symbol, Strategy] = {
            "PICNIC_BASKET1": Basket1Strategy("PICNIC_BASKET1", self.limits["PICNIC_BASKET1"]),
            "PICNIC_BASKET2": Basket2Strategy("PICNIC_BASKET2", self.limits["PICNIC_BASKET2"]),
        }

        # Volcanos
        self.iv_history_rock = []
        self.iv_window_rock = 20
        self.penelope_score = {product: 0 for product in self.voucher_products}
        self.penelope_trade_history = {product: [] for product in self.voucher_products}
        self.penelope_threshold = 3  # how many bad trades until we exploit

        # Macaron
        self.macaron_limit = 75
        self.conversion_limit = 10
        self.estimated_fee = 2.0
        self.smoothing = 0.2
        self.spread = 2  # spread around midprice for market making


    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.limits[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume
    
    def take_best_orders_with_adverse(
            self,
            product: str,
            fair_value: int,
            take_width: float,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            adverse_volume: int,
        ) -> (int, int):

            position_limit = self.limits[product]
            if len(order_depth.sell_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_amount = -1 * order_depth.sell_orders[best_ask]
                if abs(best_ask_amount) <= adverse_volume:
                    if best_ask <= fair_value - take_width:
                        quantity = min(
                            best_ask_amount, position_limit - position
                        )  # max amt to buy
                        if quantity > 0:
                            orders.append(Order(product, best_ask, quantity))
                            buy_order_volume += quantity
                            order_depth.sell_orders[best_ask] += quantity
                            if order_depth.sell_orders[best_ask] == 0:
                                del order_depth.sell_orders[best_ask]

            if len(order_depth.buy_orders) != 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid]
                if abs(best_bid_amount) <= adverse_volume:
                    if best_bid >= fair_value + take_width:
                        quantity = min(
                            best_bid_amount, position_limit + position
                        )  # should be the max we can sell
                        if quantity > 0:
                            orders.append(Order(product, best_bid, -1 * quantity))
                            sell_order_volume += quantity
                            order_depth.buy_orders[best_bid] -= quantity
                            if order_depth.buy_orders[best_bid] == 0:
                                del order_depth.buy_orders[best_bid]

            return buy_order_volume, sell_order_volume
    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.limits[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.limits[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.limits[product] - (position + buy_order_volume)
        sell_quantity = self.limits[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def take_orders_kelp(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
# Squid
    def squid_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUID]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUID]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("squid_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["squid_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("squid_last_price", None) != None:
                last_price = traderObject["squid_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.SQUID]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["squid_last_price"] = mmmid_price
            return fair
        return None

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2

    def get_best_bid_ask(self, order_depth):
        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)
        return best_bid, best_ask
    
    def update_ema(self, product: str, mid: float):
        if product not in self.ema:
            self.ema[product] = mid
        else:
            self.ema[product] = ALPHA * mid + (1 - ALPHA) * self.ema[product]
    
    def update_price_history(self, product: str, mid: float):
        if product not in self.price_history:
            self.price_history[product] = []
        self.price_history[product].append(mid)
        if len(self.price_history[product]) > VOL_WINDOW:
            self.price_history[product].pop(0)
    
    
    def get_volatility(self, product: str):
        if product not in self.price_history or len(self.price_history[product]) < 2:
            return 1
        hist = self.price_history[product]
        mean = sum(hist) / len(hist)
        return (sum((x - mean) ** 2 for x in hist) / len(hist)) ** 0.5

    def update_volatility(self, product: str, mid: float):
        window = VOL_WINDOW
        if product not in self.price_history_volatility:
            self.price_history_volatility[product] = []
        self.price_history_volatility[product].append(mid)
        if len(self.price_history_volatility[product]) > window:
            self.price_history_volatility[product].pop(0)
        if len(self.price_history_volatility[product]) >= 2:
            diffs = [(p - self.ema[product]) ** 2 for p in self.price_history_volatility[product]]
            self.volatility[product] = (sum(diffs) / len(diffs)) ** 0.5
        else:
            self.volatility[product] = 1.0  # Avoid divide-by-zero
    
    def compute_etf_fair(self, state: TradingState, basket: str):
        mids = {}
        for prod in [CROISSANTS, JAMS, DJEMBE]:
            if prod in state.order_depths:
                mids[prod] = self.get_mid_price(state.order_depths[prod])
        if CROISSANTS not in mids or JAMS not in mids:
            return None
        if basket == PICNIC_BASKET1:
            return 6 * mids[CROISSANTS] + 3 * mids[JAMS] + (mids.get(DJEMBE, 0) or 0)
        elif basket == PICNIC_BASKET2:
            return 4 * mids[CROISSANTS] + 2 * mids[JAMS]
        return None


    def take_orders_resin(self, state, product, fair_price, vol):
        k = K[product]
        positions = state.position
        order_depth = state.order_depths[product]
        position = positions.get(product, 0)
        limit = self.limits[product]

        orders = []

        # Buy logic
        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if ask_price < fair_price - k*vol:
                buy_volume = min(abs(ask_volume), limit - position, MIN_VOLUME)
                if buy_volume > 0:
                    orders.append(Order(product, ask_price, buy_volume))
                    position += buy_volume
            else:
                break

        # Sell logic
        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > fair_price + k*vol:
                sell_volume = min(abs(bid_volume), limit + position, MIN_VOLUME)
                if sell_volume > 0:
                    orders.append(Order(product, bid_price, -sell_volume))
                    position -= sell_volume
            else:
                break
        return orders
    
    def make_market(self, state, product, fair_price, edge):
        positions = state.position
        orders = []
        buy_price = int(fair_price - edge)
        sell_price = int(fair_price + edge)
        limit = self.limits[product]
        position = positions.get(product, 0)

        buy_volume = min(MIN_VOLUME, limit - position)
        sell_volume = min(MIN_VOLUME, position + limit)

        if buy_volume > 0:
            orders.append(Order(product, buy_price, buy_volume))

        if sell_volume > 0:
            orders.append(Order(product, sell_price, -sell_volume))
        return orders

    # Options
    def norm_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def bs_call_price(self, S, K, T, sigma, r=0):
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)

    def bs_call_delta(self, S, K, T, sigma, r=0):
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return self.norm_cdf(d1)

    def implied_vol(self, S, K, T, market_price, r=0):
        low, high = 1e-6, 3.0
        for _ in range(100):
            mid = (low + high) / 2
            price = self.bs_call_price(S, K, T, mid, r)
            if abs(price - market_price) < 1e-6:
                return mid
            if price > market_price:
                high = mid
            else:
                low = mid
        return mid

    def get_mid_price(self, order_depth: OrderDepth, traderData, product: str):
        if not order_depth:
            return None
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid = (best_bid + best_ask) / 2
            traderData.setdefault('price_history', {}).setdefault(product, []).append(mid)
            return mid
        elif 'price_history' in traderData and product in traderData['price_history']:
            return traderData['price_history'][product][-1]
        else:
            return None
    
    def track_penelope_trade(self, product, price, theo_price, is_buy):
        """Update penelope's score based on whether his trade looks dumb"""
        if theo_price == 0:
            margin = 0 
        else:
            margin = abs(price - theo_price) / theo_price

        if is_buy and price > theo_price * 1.02:
            self.penelope_score[product] += 1
            self.penelope_trade_history[product].append(("bad_buy", margin))
        elif not is_buy and price < theo_price * 0.98:
            self.penelope_score[product] += 1
            self.penelope_trade_history[product].append(("bad_sell", margin))
        else:
            self.penelope_score[product] = max(0, self.penelope_score[product] - 1)

    def mean_reversion_voucher_orders(
        self,
        voucher: str,
        order_depth: OrderDepth,
        position: int,
        traderData,
        rock_mid: float
    ):
        history = traderData.setdefault('price_history', {}).setdefault(voucher, [])
        mid_price = self.get_mid_price(order_depth, traderData, voucher)

        if mid_price is None:
            return None, None

        # Maintain rolling window
        window = self.params[voucher]['std_window']
        if len(history) > window:
            history.pop(0)
        if len(history) < window:
            return None, None

        mean_price = np.mean(history)
        std_price = np.std(history)

        zscore = (mid_price - mean_price) / std_price if std_price > 1e-6 else 0

        threshold = self.params[voucher]['zscore_threshold']

        orders, quote = [], []

        # Mean reversion logic
        if zscore > threshold and position > -self.limits[voucher]:
            if len(order_depth.buy_orders) == 0:
                return None, None
            best_bid = max(order_depth.buy_orders.keys())
            qty = min(self.limits[voucher] + position, order_depth.buy_orders[best_bid])
            orders.append(Order(voucher, best_bid, -qty))

        elif zscore < -threshold and position < self.limits[voucher]:
            if len(order_depth.sell_orders) == 0:
                return None, None
            best_ask = min(order_depth.sell_orders.keys())
            qty = min(self.limits[voucher] - position, order_depth.sell_orders[best_ask])
            orders.append(Order(voucher, best_ask, qty))

        # Delta for hedging (simplified: use Black-Scholes)
        strike = self.params[voucher]["strike"]
        time = self.params[voucher]["starting_time_to_expiry"]
        vol = self.params[voucher]["mean_volatility"]
        delta = BlackScholes.delta(rock_mid, strike, time, vol)
        delta = None
        return orders, delta
# Macarons
    def macarons_implied_bid_ask(
        self,
        observation: ConversionObservation,
    ) -> (float, float):
        return observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1, observation.askPrice + observation.importTariff + observation.transportFees

    def macarons_adap_edge(
        self,
        timestamp: int,
        curr_edge: float,
        position: int,
        traderObject: dict
    ) -> float: 
        if timestamp == 0:
            traderObject["MACARONS"]["curr_edge"] = self.params[Product.MACARONS]["init_make_edge"]
            return self.params[Product.MACARONS]["init_make_edge"]

        # Timestamp not 0
        traderObject["MACARONS"]["volume_history"].append(abs(position))
        if len(traderObject["MACARONS"]["volume_history"]) > self.params[Product.MACARONS]["volume_avg_timestamp"]:
            traderObject["MACARONS"]["volume_history"].pop(0)

        if len(traderObject["MACARONS"]["volume_history"]) < self.params[Product.MACARONS]["volume_avg_timestamp"]:
            return curr_edge
        elif not traderObject["MACARONS"]["optimized"]:
            volume_avg = np.mean(traderObject["MACARONS"]["volume_history"])

            # Bump up edge if consistently getting lifted full size
            if volume_avg >= self.params[Product.MACARONS]["volume_bar"]:
                traderObject["MACARONS"]["volume_history"] = [] # clear volume history if edge changed
                traderObject["MACARONS"]["curr_edge"] = curr_edge + self.params[Product.MACARONS]["step_size"]
                return curr_edge + self.params[Product.MACARONS]["step_size"]

            # Decrement edge if more cash with less edge, included discount
            elif self.params[Product.MACARONS]["dec_edge_discount"] * self.params[Product.MACARONS]["volume_bar"] * (curr_edge - self.params[Product.MACARONS]["step_size"]) > volume_avg * curr_edge:
                if curr_edge - self.params[Product.MACARONS]["step_size"] > self.params[Product.MACARONS]["min_edge"]:
                    traderObject["MACARONS"]["volume_history"] = [] # clear volume history if edge changed
                    traderObject["MACARONS"]["curr_edge"] = curr_edge - self.params[Product.MACARONS]["step_size"]
                    traderObject["MACARONS"]["optimized"] = True
                    return curr_edge - self.params[Product.MACARONS]["step_size"]
                else:
                    traderObject["MACARONS"]["curr_edge"] = self.params[Product.MACARONS]["min_edge"]
                    return self.params[Product.MACARONS]["min_edge"]

        traderObject["MACARONS"]["curr_edge"] = curr_edge
        return curr_edge

    def macarons_arb_take(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        adap_edge: float,
        position: int
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.limits[Product.MACARONS]
        buy_order_volume = 0
        sell_order_volume = 0

        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation)                                                                                                                                                                    

        buy_quantity = position_limit - position
        sell_quantity = position_limit + position

        ask = implied_ask + adap_edge

        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6

        if aggressive_ask > implied_ask:
            ask = aggressive_ask

        edge = (ask - implied_ask) * self.params[Product.MACARONS]["make_probability"]

        for price in sorted(list(order_depth.sell_orders.keys())):
            if price > implied_bid - edge:
                break

            if price < implied_bid - edge:
                quantity = min(abs(order_depth.sell_orders[price]), buy_quantity) # max amount to buy
                if quantity > 0:
                    orders.append(Order(Product.MACARONS, round(price), quantity))
                    buy_order_volume += quantity

        for price in sorted(list(order_depth.buy_orders.keys()), reverse=True):
            if price < implied_ask + edge:
                break

            if price > implied_ask + edge:
                quantity = min(abs(order_depth.buy_orders[price]), sell_quantity) # max amount to sell
                if quantity > 0:
                    orders.append(Order(Product.MACARONS, round(price), -quantity))
                    sell_order_volume += quantity

        return orders, buy_order_volume, sell_order_volume

    def macarons_arb_clear(
        self,
        position: int
    ) -> int:
        conversions = -position
        return conversions

    def macarons_arb_make(
        self,
        order_depth: OrderDepth,
        observation: ConversionObservation,
        position: int,
        edge: float,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        position_limit = self.limits[Product.MACARONS]

        # Implied Bid = observation.bidPrice - observation.exportTariff - observation.transportFees - 0.1
        # Implied Ask = observation.askPrice + observation.importTariff + observation.transportFees
        implied_bid, implied_ask = self.macarons_implied_bid_ask(observation)

        bid = implied_bid - edge
        ask = implied_ask + edge

        # ask = foreign_mid - 1.6 best performance so far
        foreign_mid = (observation.askPrice + observation.bidPrice) / 2
        aggressive_ask = foreign_mid - 1.6 # Aggressive ask

        # don't lose money
        if aggressive_ask >= implied_ask + self.params[Product.MACARONS]['min_edge']:
            ask = aggressive_ask
            print("AGGRESSIVE")
            print(f"ALGO ASK: {round(ask)}")
            print(f"ALGO BID: {round(bid)}")
        else:
            print(f"ALGO ASK: {round(ask)}")
            print(f"ALGO BID: {round(bid)}")

        filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 40]
        filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 25]

        # If we're not best level, penny until min edge
        if len(filtered_ask) > 0 and ask > filtered_ask[0]:
            if filtered_ask[0] - 1 > implied_ask:
                ask = filtered_ask[0] - 1
            else:
                ask = implied_ask + edge
        if len(filtered_bid) > 0 and  bid < filtered_bid[0]:
            if filtered_bid[0] + 1 < implied_bid:
                bid = filtered_bid[0] + 1
            else:
                bid = implied_bid - edge

        print(f"IMPLIED_BID: {implied_bid}")
        print(f"IMPLIED_ASK: {implied_ask}")
        print(f"FOREIGN ASK: {observation.askPrice}")
        print(f"FOREIGN BID: {observation.bidPrice}")

        best_bid = min(order_depth.buy_orders.keys())
        best_ask = max(order_depth.sell_orders.keys())

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(Product.MACARONS, round(bid), buy_quantity))  # Buy order

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(Product.MACARONS, round(ask), -sell_quantity))  # Sell order

        return orders, buy_order_volume, sell_order_volume


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        traderObject = {}

        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        result = {}
        conversions = 0

        order_depths = state.order_depths

        # RESIN
        if RESIN in order_depths:
            edge = 3
            fair_value = 10000
            take_orders_resin = self.take_orders_resin(state, RESIN, fair_value, 0)
            make_orders = self.make_market(state, RESIN, fair_value, edge)
            # make_orders = []
            orders = take_orders_resin + make_orders
            result[RESIN] = orders

        # SQUID
        if Product.SQUID in self.params and Product.SQUID in state.order_depths:
            SQUID_position = (
                state.position[Product.SQUID]
                if Product.SQUID in state.position
                else 0
            )
            SQUID_fair_value = self.squid_fair_value(
                state.order_depths[Product.SQUID], traderObject
            )
            SQUID_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.SQUID,
                    state.order_depths[Product.SQUID],
                    SQUID_fair_value,
                    self.params[Product.SQUID]["take_width"],
                    SQUID_position,
                    self.params[Product.SQUID]["prevent_adverse"],
                    self.params[Product.SQUID]["adverse_volume"],
                )
            )
            SQUID_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID,
                    state.order_depths[Product.SQUID],
                    SQUID_fair_value,
                    self.params[Product.SQUID]["clear_width"],
                    SQUID_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            SQUID_make_orders, _, _ = self.make_orders(
                Product.SQUID,
                state.order_depths[Product.SQUID],
                SQUID_fair_value,
                SQUID_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID]["disregard_edge"],
                self.params[Product.SQUID]["join_edge"],
                self.params[Product.SQUID]["default_edge"],
            )
            result[Product.SQUID] = (
                SQUID_take_orders + SQUID_clear_orders + SQUID_make_orders
            )

        # KELP
        if KELP in self.params and KELP in state.order_depths:
            KELP_position = (
                state.position[KELP]
                if KELP in state.position
                else 0
            )
            KELP_fair_value = self.kelp_fair_value(
                state.order_depths[KELP], traderObject
            )
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders_kelp(
                    KELP,
                    state.order_depths[KELP],
                    KELP_fair_value,
                    self.params[KELP]["take_width"],
                    KELP_position,
                    self.params[KELP]["prevent_adverse"],
                    self.params[KELP]["adverse_volume"],
                )
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    KELP,
                    state.order_depths[KELP],
                    KELP_fair_value,
                    self.params[KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            KELP_make_orders, _, _ = self.make_orders(
                KELP,
                state.order_depths[KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[KELP]["disregard_edge"],
                self.params[KELP]["join_edge"],
                self.params[KELP]["default_edge"],
            )
            result[KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )
        
        # ETF

        old_trader_data = jsonpickle.decode(state.traderData) if state.traderData else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data[symbol])

            if symbol in state.order_depths and state.order_depths[symbol].buy_orders and state.order_depths[symbol].sell_orders:
                strategy_orders, strategy_conversions = strategy.run(state)
                result[symbol] = strategy_orders
                conversions += strategy_conversions

            new_trader_data[symbol] = strategy.save()

        # Volcanos
        products = ROCK_VOUCHERS

        if not state.traderData or 'price_history' not in state.traderData:
              traderObject['price_history'] = {p: [] for p in products}
        else:
            # traderData = jsonpickle.decode(state.traderData)
            for p in products:
                traderObject['price_history'].setdefault(p, [])

        all_orders = {}
        all_deltas = {}

        rock_order_depth = state.order_depths.get(ROCK)
        rock_position = state.position.get(ROCK, 0)
        rock_mid = self.get_mid_price(rock_order_depth, traderObject, ROCK)

        for voucher in ROCK_VOUCHERS:
            order_depth = state.order_depths.get(voucher)
            if not order_depth:
                continue
            position = state.position.get(voucher, 0)

            voucher_orders, delta = self.mean_reversion_voucher_orders(
                voucher,
                order_depth,
                position,
                traderObject,
                rock_mid
            )

            if voucher_orders:
                all_orders[voucher] = voucher_orders

            strike = int(voucher.split("_")[-1])
            day = state.timestamp // 10000
            TTE = max(0, 7 - day) / 7 

            iv_samples = []
            if order_depth.buy_orders and order_depth.sell_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_ask = min(order_depth.sell_orders.keys())
                    voucher_price = (best_bid + best_ask) / 2

                    implied_vol = self.implied_vol(rock_mid, strike, TTE, voucher_price)
                    iv_samples.append(implied_vol)

                    theo_price = self.bs_call_price(rock_mid, strike, TTE, implied_vol)
                    delta = self.bs_call_delta(rock_mid, strike, TTE, implied_vol)

                    iv_mean = np.mean(self.iv_history_rock) if len(self.iv_history_rock) >= self.iv_window_rock else np.mean(iv_samples) if iv_samples else 0

                    trade_size = max(1, round(5 * TTE * iv_mean))
                    position = state.position.get(voucher, 0)
            
                    # Simulated penelope trade detection (assume he trades mid)
                    for trade in state.market_trades.get(voucher, []):
                        if trade.buyer == "Penelope": 
                            self.track_penelope_trade(voucher, voucher_price, theo_price, is_buy=True)
                        if trade.seller == "Penelope": 
                            self.track_penelope_trade(voucher, voucher_price, theo_price, is_buy=False)

                    penelope_is_wild = self.penelope_score[voucher] >= self.penelope_threshold
                    max_position = 200 * (1 / iv_mean)

                    # Buy if undervalued OR penelope was a bad seller
                    if (implied_vol < iv_mean * 0.95 or voucher_price < theo_price * 0.98) and penelope_is_wild:
                        if position + trade_size <= max_position:
                            all_orders[voucher].append(Order(voucher, best_ask, trade_size))
                            all_deltas[voucher] += trade_size * delta

                    # Sell if overvalued OR penelope was a bad buyer
                    elif (implied_vol > iv_mean * 1.05 or voucher_price > theo_price * 1.02) and penelope_is_wild:
                        if position - trade_size >= -max_position:
                            all_orders[voucher].append(Order(voucher, best_bid, -trade_size))
                            all_deltas[voucher] -= trade_size * delta


            if delta is not None:
                all_deltas[voucher] = (position, delta)

        # Hedge ROCK based on total delta from vouchers
        total_hedge_qty = 0
        for voucher, (position, delta) in all_deltas.items():
            total_hedge_qty += -delta * position

        hedge_qty = round(total_hedge_qty - rock_position)

        hedge_orders = []
        if hedge_qty > 0:
            best_ask = min(rock_order_depth.sell_orders.keys())
            qty = min(hedge_qty, self.limits[ROCK] - rock_position)
            hedge_orders.append(Order(ROCK, best_ask, qty))
        elif hedge_qty < 0:
            best_bid = max(rock_order_depth.buy_orders.keys())
            qty = min(-hedge_qty, self.limits[ROCK] + rock_position)
            hedge_orders.append(Order(ROCK, best_bid, -qty))

        if hedge_orders:
            all_orders[ROCK] = hedge_orders
        for product, orders in all_orders.items():
            if orders:
                result[product] = orders

        # Macaron
        conversions = 0

        if Product.MACARONS in self.params and Product.MACARONS in state.order_depths:
            if "MACARONS" not in traderObject:
                traderObject["MACARONS"] = {"curr_edge": self.params[Product.MACARONS]["init_make_edge"], "volume_history": [], "optimized": False}
            macarons_position = (
                state.position[Product.MACARONS]
                if Product.MACARONS in state.position
                else 0
            )
            print(f"MACARONS POSITION: {macarons_position}")

            conversions = self.macarons_arb_clear(
                macarons_position
            )

            adap_edge = self.macarons_adap_edge(
                state.timestamp,
                traderObject["MACARONS"]["curr_edge"],
                macarons_position,
                traderObject,
            )

            macarons_position = 0

            macarons_take_orders, buy_order_volume, sell_order_volume = self.macarons_arb_take(
                state.order_depths[Product.MACARONS],
                state.observations.conversionObservations[Product.MACARONS],
                adap_edge,
                macarons_position,
            )

            macarons_make_orders, _, _ = self.macarons_arb_make(
                state.order_depths[Product.MACARONS],
                state.observations.conversionObservations[Product.MACARONS],
                macarons_position,
                adap_edge,
                buy_order_volume,
                sell_order_volume
            )

            result[Product.MACARONS] = (
                macarons_take_orders + macarons_make_orders
            )

        traderData = jsonpickle.encode(traderObject)
        
        return result, conversions, traderData