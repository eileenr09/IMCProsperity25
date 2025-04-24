
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

PICNIC_BASKET1 = "PICNIC_BASKET1"  # 6 CROISSANTS, 3 JAMS, 1 DJEMBE
PICNIC_BASKET2 = "PICNIC_BASKET2"  # 4 CROISSANTS, 2 JAMS
CROISSANTS = "CROISSANTS"
JAMS = "JAMS"
DJEMBE = "DJEMBE"

LIMIT_BSK1 = 60
LIMIT_BSK2 = 100

THRESHOLD_HIGH = 0.8
THRESHOLD_LOW = -0.8
PRICE_OFFSET = 1  # Shift bid/ask slightly to increase chance of execution
ALPHA = 0.1
VOL_WINDOW = 20

class Trader:
    def __init__(self):
        self.ema = {}
        self.price_history = {}

    def get_mid_price(self, order_depth: OrderDepth):
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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for product in [PICNIC_BASKET1, PICNIC_BASKET2]:
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            mid = self.get_mid_price(order_depth)
            if mid is None:
                continue

            self.update_ema(product, mid)
            self.update_price_history(product, mid)
            vol = self.get_volatility(product)

            fair = self.compute_etf_fair(state, product)
            if fair is None:
                continue

            zscore = (mid - fair) / vol if vol > 0 else 0
            orders = []
            limit = LIMIT_BSK1 if product == PICNIC_BASKET1 else LIMIT_BSK2
            pos = state.position.get(product, 0)

            # Buy if underpriced relative to ETF fair value
            if zscore < THRESHOLD_LOW:
                best_ask = min(order_depth.sell_orders)
                buy_price = best_ask + PRICE_OFFSET
                buy_volume = min(abs(order_depth.sell_orders[best_ask]), limit - pos)
                if buy_volume > 0:
                    orders.append(Order(product, buy_price, buy_volume))

            # Sell if overpriced
            if zscore > THRESHOLD_HIGH:
                best_bid = max(order_depth.buy_orders)
                sell_price = best_bid - PRICE_OFFSET
                sell_volume = min(abs(order_depth.buy_orders[best_bid]), limit + pos)
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume))

            result[product] = orders

        traderData = "SAMPLE"
        conversions = 1 
        
        return result, conversions, traderData

