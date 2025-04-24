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

class Trader:
    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2

    def run(self, state: TradingState):
        result = {}
        product_prices = {}
        
        for product, order_depth in state.order_depths.items():
            mid = self.get_mid_price(order_depth)
            if mid is not None:
                product_prices[product] = mid
        
        position = {p: state.position.get(p, 0) for p in [PICNIC_BASKET1, PICNIC_BASKET2]}

        # --- BASKET 1 ---
        if all(p in product_prices for p in [PICNIC_BASKET1, CROISSANTS, JAMS, DJEMBE]):
            pb1_price = product_prices[PICNIC_BASKET1]
            fair_pb1 = 6 * product_prices[CROISSANTS] + 3 * product_prices[JAMS] + product_prices[DJEMBE]
            spread_pb1 = pb1_price - fair_pb1

            orders_pb1 = []

            # Overpriced: Sell PB1
            if spread_pb1 > THRESHOLD_HIGH and position[PICNIC_BASKET1] > -LIMIT_BSK1:
                qty = min(10, LIMIT_BSK1 + position[PICNIC_BASKET1])
                price = fair_pb1 - PRICE_OFFSET
                print(f"SELL PB1: {qty} @ {price} (Spread: {spread_pb1:.2f})")
                orders_pb1.append(Order(PICNIC_BASKET1, int(price), -qty))

            # Underpriced: Buy PB1
            if spread_pb1 < THRESHOLD_LOW and position[PICNIC_BASKET1] < LIMIT_BSK1:
                qty = min(10, LIMIT_BSK1 - position[PICNIC_BASKET1])
                price = fair_pb1 + PRICE_OFFSET
                print(f"BUY PB1: {qty} @ {price} (Spread: {spread_pb1:.2f})")
                orders_pb1.append(Order(PICNIC_BASKET1, int(price), qty))

            if orders_pb1:
                result[PICNIC_BASKET1] = orders_pb1

        # --- BASKET 2 ---
        if all(p in product_prices for p in [PICNIC_BASKET2, CROISSANTS, JAMS]):
            pb2_price = product_prices[PICNIC_BASKET2]
            fair_pb2 = 4 * product_prices[CROISSANTS] + 2 * product_prices[JAMS]
            spread_pb2 = pb2_price - fair_pb2

            orders_pb2 = []

            # Overpriced: Sell PB2
            if spread_pb2 > THRESHOLD_HIGH and position[PICNIC_BASKET2] > -LIMIT_BSK2:
                qty = min(10, LIMIT_BSK2 + position[PICNIC_BASKET2])
                price = fair_pb2 - PRICE_OFFSET
                print(f"SELL PB2: {qty} @ {price} (Spread: {spread_pb2:.2f})")
                orders_pb2.append(Order(PICNIC_BASKET2, int(price), -qty))

            # Underpriced: Buy PB2
            if spread_pb2 < THRESHOLD_LOW and position[PICNIC_BASKET2] < LIMIT_BSK2:
                qty = min(10, LIMIT_BSK2 - position[PICNIC_BASKET2])
                price = fair_pb2 + PRICE_OFFSET
                print(f"BUY PB2: {qty} @ {price} (Spread: {spread_pb2:.2f})")
                orders_pb2.append(Order(PICNIC_BASKET2, int(price), qty))

            if orders_pb2:
                result[PICNIC_BASKET2] = orders_pb2

        traderData = "SAMPLE"
        
        conversions = 1 
        
        return result, conversions, traderData
