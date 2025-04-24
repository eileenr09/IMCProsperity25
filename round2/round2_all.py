
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"

RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID = "SQUID_INK"

ROUND1_PRODUCTS = [RESIN, KELP, SQUID]
ROUND1_LIMIT = 50

LIMIT_BSK1 = 60
LIMIT_BSK2 = 100

LIMITS = {RESIN: 50, KELP: 50, SQUID: 50, PICNIC_BASKET1: LIMIT_BSK1, PICNIC_BASKET2: LIMIT_BSK2}

ALPHA = 0.1  # EMA smoothing factor

SQUID_MA_WINDOW = 5

VOL_WINDOW = {
    SQUID: 30,
    PICNIC_BASKET1: 20,
    PICNIC_BASKET2: 20,
    KELP: 20,
}

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
    def __init__(self):
        self.ema = {}
        self.ma = {}
        self.volatility = {}
        self.price_history_vol = {}
        self.price_history_ma = {}

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

    def update_ma(self, product: str, mid: float):
        window = SQUID_MA_WINDOW
        if product not in self.price_history_ma:
            self.price_history_ma[product] = []
        self.price_history_ma[product].append(mid)
        if len(self.price_history_ma[product]) > window:
            self.price_history_ma[product].pop(0)
        self.ma[product] = sum(self.price_history_ma[product]) / len(self.price_history_ma[product])

    def update_volatility(self, product: str, mid: float):
        window = VOL_WINDOW[product]
        if product not in self.price_history_vol:
            self.price_history_vol[product] = []
        self.price_history_vol[product].append(mid)
        if len(self.price_history_vol[product]) > window:
            self.price_history_vol[product].pop(0)
        if len(self.price_history_vol[product]) >= 2:
            diffs = [(p - self.ema[product]) ** 2 for p in self.price_history_vol[product]]
            self.volatility[product] = (sum(diffs) / len(diffs)) ** 0.5
        else:
            self.volatility[product] = 1.0  # Avoid divide-by-zero
    
    def take_orders(self, state, product, fair_price, vol):
        k = K[product]
        positions = state.position
        order_depth = state.order_depths[product]
        position = positions.get(product, 0)
        limit = LIMITS[product]

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
        limit = LIMITS[product]
        position = positions.get(product, 0)

        buy_volume = min(MIN_VOLUME, limit - position)
        sell_volume = min(MIN_VOLUME, position + limit)

        if buy_volume > 0:
            orders.append(Order(product, buy_price, buy_volume))

        if sell_volume > 0:
            orders.append(Order(product, sell_price, -sell_volume))
        return orders

    def trade_mean_reversion(self, product: str, state: TradingState) -> List[Order]:
        order_depth = state.order_depths[product]
        mid = self.get_mid_price(order_depth)
        if mid is None:
            return []

        self.update_ema(product, mid)
        self.update_ma(product, mid)
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

        order_depths = state.order_depths

        # RESIN
        if RESIN in order_depths:
            edge = 3
            fair_value = 10000
            take_orders = self.take_orders(state, RESIN, fair_value, 0)
            make_orders = self.make_market(state, RESIN, fair_value, edge)
            # make_orders = []
            orders = take_orders + make_orders
            result[RESIN] = orders

        # SQUID
        if SQUID in order_depths:
            order_depth = state.order_depths[SQUID]
            div = 91000
            if state.timestamp < div:
                if SQUID in state.order_depths:
                    squid_orders = self.trade_mean_reversion(SQUID, state)
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
                elif position < 0:
                    result[SQUID] = [(Order(SQUID, best_ask, -position))]
            elif state.timestamp > div:
                result[SQUID] = []
            
        for product in [PICNIC_BASKET1, PICNIC_BASKET2, KELP]:
            if product not in order_depths:
                continue
            order_depth = order_depths[product]

            mid = self.get_mid_price(order_depth)
            if mid is None:
                continue

            self.update_ema(product, mid)
            self.update_volatility(product, mid)

            fair_price = self.ema[product]
            vol = self.volatility[product]

            take_orders = self.take_orders(state, product, fair_price, vol)
            # make_orders = self.make_market(state, product, fair_price, SPREAD_EDGE)
            result[product] = take_orders 

        traderData = "SAMPLE"
        conversions = 1 
        
        return result, conversions, traderData
