from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math

strikes = [9500, 9750, 10000, 10250, 10500]

ROCK = "VOLCANIC_ROCK"
ROCK_VOUCHERS = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in strikes]
    

PARAMS = {
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
}

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

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.strikes = strikes
        self.voucher_products = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in self.strikes]
        self.rock_product = "VOLCANIC_ROCK"


        self.limits = {
            ROCK: 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }
        self.iv_history_rock = []
        self.iv_window_rock = 20
        self.penelope_score = {product: 0 for product in self.voucher_products}
        self.penelope_trade_history = {product: [] for product in self.voucher_products}
        self.penelope_threshold = 3  # how many bad trades until we exploit

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

    def get_mid_price(self, order_depth: OrderDepth, traderData: Dict[str, Any], product: str):
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
        traderData: Dict[str, Any],
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

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        products = ROCK_VOUCHERS
        
        if not state.traderData or 'price_history' not in state.traderData:
              traderData = {'price_history': {p: [] for p in products}}
        else:
            traderData = jsonpickle.decode(state.traderData)
            for p in products:
                traderData['price_history'].setdefault(p, [])

        all_orders = {}
        all_deltas = {}

        rock_order_depth = state.order_depths.get(ROCK)
        rock_position = state.position.get(ROCK, 0)
        rock_mid = self.get_mid_price(rock_order_depth, traderData, ROCK)

        for voucher in ROCK_VOUCHERS:
            order_depth = state.order_depths.get(voucher)
            if not order_depth:
                continue
            position = state.position.get(voucher, 0)

            voucher_orders, delta = self.mean_reversion_voucher_orders(
                voucher,
                order_depth,
                position,
                traderData,
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

        result = {product: orders for product, orders in all_orders.items() if orders}

        return result, conversions, jsonpickle.encode(traderData)

