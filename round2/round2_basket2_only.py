
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"

LIMIT_BSK1 = 60
LIMIT_BSK2 = 100

ALPHA = 0.1  # EMA smoothing factor
VOL_WINDOW = 20  # Number of ticks to track for dynamic threshold
MIN_VOLUME = 5  # Minimum volume to trade
BASE_SPREAD = 1.0  # Base price offset to increase fill chance

class Trader:
    def __init__(self):
        self.ema = {}
        self.volatility = {}
        self.price_history = {}

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2

    def update_ema(self, product: str, mid: float):
        if product not in self.ema:
            self.ema[product] = mid
        else:
            self.ema[product] = ALPHA * mid + (1 - ALPHA) * self.ema[product]

    def update_volatility(self, product: str, mid: float):
        if product not in self.price_history:
            self.price_history[product] = []
        self.price_history[product].append(mid)
        if len(self.price_history[product]) > VOL_WINDOW:
            self.price_history[product].pop(0)
        if len(self.price_history[product]) >= 2:
            diffs = [(p - self.ema[product]) ** 2 for p in self.price_history[product]]
            self.volatility[product] = (sum(diffs) / len(diffs)) ** 0.5
        else:
            self.volatility[product] = 1.0  # Avoid divide-by-zero

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        positions = state.position
        order_depths = state.order_depths

        for product in [PICNIC_BASKET2]:
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
            position = positions.get(product, 0)
            limit = LIMIT_BSK1 if product == PICNIC_BASKET1 else LIMIT_BSK2

            orders = []

            # Buy logic
            for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                if ask_price < fair_price - vol:
                    buy_volume = min(abs(ask_volume), limit - position, MIN_VOLUME)
                    if buy_volume > 0:
                        orders.append(Order(product, ask_price, buy_volume))
                        position += buy_volume
                else:
                    break

            # Sell logic
            for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                if bid_price > fair_price + vol:
                    sell_volume = min(abs(bid_volume), limit + position, MIN_VOLUME)
                    if sell_volume > 0:
                        orders.append(Order(product, bid_price, -sell_volume))
                        position -= sell_volume
                else:
                    break

            result[product] = orders

        traderData = "SAMPLE"
        conversions = 1 
        
        return result, conversions, traderData
