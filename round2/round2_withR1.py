
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import math
import numpy as np

# --- Constants ---
PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"
RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"

LIMITS = {
    PICNIC_BASKET1: 60,
    PICNIC_BASKET2: 100,
    RAINFOREST_RESIN: 50,
    KELP: 50,
    SQUID_INK: 50
}

ALPHA = 0.1  # EMA smoothing
VOL_WINDOW = 20
MIN_VOLUME = 5
BASE_SPREAD = 1.0


class Trader:
    def __init__(self):
        self.ema = {}
        self.volatility = {}
        self.price_history = {}
        self.resin_fair_value = 10000
        self.kelp_history = []
        self.squid_history = []

    def get_mid_price(self, order_depth: OrderDepth):
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders)
        best_bid = max(order_depth.buy_orders)
        return (best_ask + best_bid) / 2

    def update_ema(self, product, price):
        if product not in self.ema:
            self.ema[product] = price
        else:
            self.ema[product] = ALPHA * price + (1 - ALPHA) * self.ema[product]

    def update_volatility(self, product, price):
        if product not in self.price_history:
            self.price_history[product] = []
        self.price_history[product].append(price)
        if len(self.price_history[product]) > VOL_WINDOW:
            self.price_history[product].pop(0)
        if len(self.price_history[product]) >= 2:
            diffs = [(p - self.ema[product]) ** 2 for p in self.price_history[product]]
            self.volatility[product] = (sum(diffs) / len(diffs)) ** 0.5
        else:
            self.volatility[product] = 1.0

    def basket_strategy(self, product, state, position, order_depth):
        mid = self.get_mid_price(order_depth)
        if mid is None:
            return []

        self.update_ema(product, mid)
        self.update_volatility(product, mid)
        fair_price = self.ema[product]
        vol = self.volatility[product]
        limit = LIMITS[product]

        orders = []

        for ask, vol_ask in sorted(order_depth.sell_orders.items()):
            if ask < fair_price - vol:
                buy_qty = min(abs(vol_ask), limit - position, MIN_VOLUME)
                if buy_qty > 0:
                    orders.append(Order(product, ask, buy_qty))
                    position += buy_qty
            else:
                break

        for bid, vol_bid in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > fair_price + vol:
                sell_qty = min(abs(vol_bid), limit + position, MIN_VOLUME)
                if sell_qty > 0:
                    orders.append(Order(product, bid, -sell_qty))
                    position -= sell_qty
            else:
                break

        return orders


    def resin_strategy(self, product, position, order_depth):
        fair = self.resin_fair_value
        limit = LIMITS[product]
        orders = []

        for ask, vol_ask in sorted(order_depth.sell_orders.items()):
            if ask < fair - 5 and vol_ask > 8:
                buy_qty = min(vol_ask, limit - position, MIN_VOLUME)
                orders.append(Order(product, ask + 1, buy_qty))
                break

        for bid, vol_bid in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > fair + 5 and vol_bid > 8:
                sell_qty = min(vol_bid, limit + position, MIN_VOLUME)
                orders.append(Order(product, bid - 1, -sell_qty))
                break

        return orders
    
        fair = self.resin_fair_value
        limit = LIMITS[product]
        orders = []

        for ask, vol_ask in sorted(order_depth.sell_orders.items()):
            if ask < fair - 4:
                buy_qty = min(abs(vol_ask), limit - position, MIN_VOLUME)
                orders.append(Order(product, ask, buy_qty))
                position += buy_qty
            else:
                break

        for bid, vol_bid in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > fair + 4:
                sell_qty = min(abs(vol_bid), limit + position, MIN_VOLUME)
                orders.append(Order(product, bid, -sell_qty))
                position -= sell_qty
            else:
                break

        return orders


    def kelp_strategy(self, product, position, order_depth):
        mid = self.get_mid_price(order_depth)
        if mid is None:
            return []

        self.kelp_history.append(mid)
        if len(self.kelp_history) > 15:
            self.kelp_history.pop(0)

        orders = []
        limit = LIMITS[product]
        if len(self.kelp_history) == 15:
            x = np.arange(15)
            y = np.array(self.kelp_history)
            slope, _ = np.polyfit(x, y, 1)

            if slope > 1.0 and position <= 0:
                best_ask = min(order_depth.sell_orders.keys())
                buy_qty = min(abs(order_depth.sell_orders[best_ask]), limit - position, MIN_VOLUME)
                orders.append(Order(product, best_ask, buy_qty))
            elif slope < -1.0 and position >= 0:
                best_bid = max(order_depth.buy_orders.keys())
                sell_qty = min(abs(order_depth.buy_orders[best_bid]), limit + position, MIN_VOLUME)
                orders.append(Order(product, best_bid, -sell_qty))
        return orders
    
        mid = self.get_mid_price(order_depth)
        if mid is None:
            return []

        self.kelp_history.append(mid)
        if len(self.kelp_history) > 10:
            self.kelp_history.pop(0)

        orders = []
        limit = LIMITS[product]
        if len(self.kelp_history) == 10:
            x = np.arange(10)
            y = np.array(self.kelp_history)
            slope, _ = np.polyfit(x, y, 1)
            if slope > 0.5:
                best_ask = min(order_depth.sell_orders.keys())
                buy_qty = min(abs(order_depth.sell_orders[best_ask]), limit - position, MIN_VOLUME)
                orders.append(Order(product, best_ask, buy_qty))
            elif slope < -0.5:
                best_bid = max(order_depth.buy_orders.keys())
                sell_qty = min(abs(order_depth.buy_orders[best_bid]), limit + position, MIN_VOLUME)
                orders.append(Order(product, best_bid, -sell_qty))
        return orders


    def squid_strategy(self, product, position, order_depth):
        mid = self.get_mid_price(order_depth)
        if mid is None:
            return []

        self.squid_history.append(mid)
        if len(self.squid_history) > 30:
            self.squid_history.pop(0)

        orders = []
        limit = LIMITS[product]
        if len(self.squid_history) >= 10:
            mean = np.mean(self.squid_history)
            std = np.std(self.squid_history)
            zscore = (mid - mean) / std if std > 0 else 0

            if zscore < -2.0:
                best_ask = min(order_depth.sell_orders.keys())
                buy_qty = min(abs(order_depth.sell_orders[best_ask]), limit - position, MIN_VOLUME)
                orders.append(Order(product, best_ask, buy_qty))
            elif zscore > 2.0:
                best_bid = max(order_depth.buy_orders.keys())
                sell_qty = min(abs(order_depth.buy_orders[best_bid]), limit + position, MIN_VOLUME)
                orders.append(Order(product, best_bid, -sell_qty))
        return orders
    
        mid = self.get_mid_price(order_depth)
        if mid is None:
            return []

        self.squid_history.append(mid)
        if len(self.squid_history) > 20:
            self.squid_history.pop(0)

        orders = []
        limit = LIMITS[product]
        if len(self.squid_history) >= 5:
            mean = np.mean(self.squid_history)
            std = np.std(self.squid_history)
            zscore = (mid - mean) / std if std > 0 else 0

            if zscore < -1.7:
                best_ask = min(order_depth.sell_orders.keys())
                buy_qty = min(abs(order_depth.sell_orders[best_ask]), limit - position, MIN_VOLUME)
                orders.append(Order(product, best_ask, buy_qty))
            elif zscore > 1.7:
                best_bid = max(order_depth.buy_orders.keys())
                sell_qty = min(abs(order_depth.buy_orders[best_bid]), limit + position, MIN_VOLUME)
                orders.append(Order(product, best_bid, -sell_qty))
        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)

            if product in [PICNIC_BASKET1, PICNIC_BASKET2]:
                result[product] = self.basket_strategy(product, state, position, order_depth)
            elif product == RAINFOREST_RESIN:
                result[product] = self.resin_strategy(product, position, order_depth)
            elif product == KELP:
                result[product] = self.kelp_strategy(product, position, order_depth)
            elif product == SQUID_INK:
                result[product] = self.squid_strategy(product, position, order_depth)

        traderData = "SAMPLE"
        conversions = 1 
        
        return result, conversions, traderData

