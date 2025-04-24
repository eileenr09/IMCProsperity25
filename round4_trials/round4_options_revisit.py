from typing import Dict, List
from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import string
import jsonpickle
import numpy as np
import math
import statistics

outer_limit = 50
LIMIT_BSK1 = 60
LIMIT_BSK2 = 100

PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"

RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID = "SQUID_INK"
CROISSANTS = "CROISSANTS"
JAMS = "JAMS"
DJEMBE = "DJEMBE"
MACARON = "MAGNIFICENT_MACARONS"

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
}

THRESHOLD_HIGH = 0.8
THRESHOLD_LOW = -0.8
PRICE_OFFSET = 1  # Shift bid/ask slightly to increase chance of execution
ALPHA = 0.1
VOL_WINDOW = 20

LIMITS = {RESIN: 50, KELP: 50, SQUID: 50, PICNIC_BASKET1: LIMIT_BSK1, PICNIC_BASKET2: LIMIT_BSK2}


K = {
    RESIN: 1,
    SQUID: 1,
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

        self.LIMIT = {RESIN: 50,
            KELP: 50,
            SQUID: 50}
        self.price_history = {
            SQUID: [],
            PICNIC_BASKET1: [],
            PICNIC_BASKET2: []
        }

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

        # Volcanos
        self.voucher_pnl = 0.0
        self.strikes = [9500, 9750, 10000, 10250, 10500]
        self.voucher_products = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in self.strikes]
        self.basic_products = ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]
        self.rock_product = "VOLCANIC_ROCK"
        self.window_size_vol = 5
        self.limit_vol = 200
        self.basic_limits = {p: 50 for p in self.basic_products}

        self.tick_count_vol = 0
        self.price_history_vol = {p: [] for p in self.basic_products}
        self.rock_prices = []  # for adaptive volatility estimation

        # Macaron
        self.macaron_limit = 75
        self.conversion_limit = 10
        self.estimated_fee = 2.0
        self.smoothing = 0.2
        self.spread = 2  # spread around midprice for market making


    def norm_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def bs_call_price(self, S, K, T, sigma, r=0):
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)

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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}

        # Volcanos
        self.tick_count_vol += 1
        # result = {}
        mid_prices = {}
        for product, depth in state.order_depths.items():
            if depth.buy_orders and depth.sell_orders:
                best_bid = max(depth.buy_orders.keys())
                best_ask = min(depth.sell_orders.keys())
                mid_prices[product] = (best_bid + best_ask) / 2

        rock_price = mid_prices.get(self.rock_product)
        if rock_price is not None:
            self.rock_prices.append(rock_price)
            if len(self.rock_prices) > 20:
                self.rock_prices.pop(0)

        day = state.timestamp // 10000

        if self.rock_product in mid_prices:
            spot = mid_prices[self.rock_product]
            T = max(1e-6, (7 - day) / 7)

            for product in self.voucher_products:
                if day == 0:
                    continue  # fully disable voucher trading on Day 0

                if product in state.order_depths and product in mid_prices:
                    strike = int(product.split('_')[-1])

                    if day == 0 and abs(strike - spot) > 250:
                        continue  # avoid risky deep OTM vouchers early

                    market_price = mid_prices[product]
                    iv = self.implied_vol(spot, strike, T, market_price)
                    fair_price = self.bs_call_price(spot, strike, T, iv)
                    mispricing = market_price - fair_price
                    order_depth = state.order_depths[product]
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                    orders = []

                    vol_multiplier = min(max(iv, 0.1), 1.0)
                    mispricing_thresh = 5 + 10 * vol_multiplier
                    qty_scale_factor = 15 / vol_multiplier
                    max_loss_per_voucher = 5000

                    if mispricing > mispricing_thresh and best_bid:
                        qty = int(self.limit_vol * min(1.0, mispricing / qty_scale_factor))
                        if qty * best_bid > max_loss_per_voucher:
                            qty = max_loss_per_voucher // best_bid
                        qty = min(qty, order_depth.buy_orders[best_bid])
                        orders.append(Order(product, best_bid, -qty))
                        self.voucher_pnl += qty * best_bid
                    elif mispricing < -mispricing_thresh and best_ask:
                        qty = int(self.limit_vol * min(1.0, abs(mispricing) / qty_scale_factor))
                        if qty * best_ask > max_loss_per_voucher:
                            qty = max_loss_per_voucher // best_ask
                        qty = min(qty, order_depth.sell_orders[best_ask])
                        orders.append(Order(product, best_ask, qty))
                        self.voucher_pnl -= qty * best_ask

                    if orders:
                        result[product] = orders
      
        traderData = jsonpickle.encode(traderObject)
        conversions = 1 
        
        return result, conversions, traderData