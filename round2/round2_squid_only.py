from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

# Product constants
PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"
RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID = "SQUID_INK"

# Limits
LIMITS = {
    RESIN: 50, KELP: 50, SQUID: 50,
    PICNIC_BASKET1: 60, PICNIC_BASKET2: 100
}

# Strategy hyperparameters
ALPHA = 0.1  # EMA smoothing factor

VOL_WINDOW = {
    SQUID: 30,
    PICNIC_BASKET1: 20,
    PICNIC_BASKET2: 20
}
MIN_VOLUME = 10
SPREAD_EDGE = 1  # Offset to increase fill chance

class Trader:
    def __init__(self):
        self.ema = {}
        self.ma = {}
        self.volatility = {}
        self.price_history_ma = {}
        self.price_history_vol = {}

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

    def update_ma(self, product: str, mid: float, MA_WINDOW):
        if product not in self.price_history_ma:
            self.price_history_ma[product] = []
        self.price_history_ma[product].append(mid)
        if len(self.price_history_ma[product]) > MA_WINDOW:
            self.price_history_ma[product].pop(0)
        self.ma[product] = sum(self.price_history_ma[product]) / len(self.price_history_ma[product])

    def update_volatility(self, product: str, mid: float):
        window = VOL_WINDOW.get(product, 20)
        if product not in self.price_history_vol:
            self.price_history_vol[product] = []
        self.price_history_vol[product].append(mid)
        if len(self.price_history_vol[product]) > window:
            self.price_history_vol[product].pop(0)
        if len(self.price_history_vol[product]) >= 2:
            diffs = [(p - self.ema[product]) ** 2 for p in self.price_history_vol[product]]
            self.volatility[product] = (sum(diffs) / len(diffs)) ** 0.5
        else:
            self.volatility[product] = 1.0

    def trade_mean_reversion(self, product: str, state: TradingState, MA_WINDOW) -> List[Order]:
        order_depth = state.order_depths[product]
        mid = self.get_mid_price(order_depth)
        if mid is None:
            return []

        self.update_ema(product, mid)
        self.update_ma(product, mid, MA_WINDOW)
        self.update_volatility(product, mid)

        fair_price = self.ma[product]
        vol = self.volatility[product]
        position = state.position.get(product, 0)
        limit = LIMITS[product]

        best_bid, best_ask = self.get_best_bid_ask(order_depth)
        orders = []

        # Buy if current price is undervalued
        if mid < fair_price - 1.5*vol:
            buy_price = min(best_ask, int(fair_price - SPREAD_EDGE)) if best_ask else int(fair_price - SPREAD_EDGE)
            buy_volume = min(MIN_VOLUME, limit - position)
            if buy_volume > 0:
                orders.append(Order(product, buy_price, buy_volume))

        # Sell if current price is overvalued
        if mid > fair_price + 1.5*vol:
            sell_price = max(best_bid, int(fair_price + SPREAD_EDGE)) if best_bid else int(fair_price + SPREAD_EDGE)
            sell_volume = min(MIN_VOLUME, position + limit)
            if sell_volume > 0:
                orders.append(Order(product, sell_price, -sell_volume))

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        order_depth = state.order_depths[SQUID]
        div = 90500
        if state.timestamp < div:
            MA_WINDOW = 5
            if SQUID in state.order_depths:
                squid_orders = self.trade_mean_reversion(SQUID, state, MA_WINDOW)
                if squid_orders:
                    result[SQUID] = squid_orders
        elif state.timestamp == div:
            if SQUID in state.position:
                position = state.position[SQUID]
            else:
                position = 0
            best_bid, best_ask = self.get_best_bid_ask(order_depth)
            if position > 0:
                result[SQUID] = [(Order(SQUID, best_bid, -position))]
            if position < 0:
                result[SQUID] = [(Order(SQUID, best_ask, -position))]
        elif state.timestamp > div:
            result[SQUID] = []
                
        traderData = "SAMPLE"
        conversions = 1
        return result, conversions, traderData
