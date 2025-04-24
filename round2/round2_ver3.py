from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

PICNIC_BASKET1 = "PICNIC_BASKET1"  # 6 CROISSANTS, 3 JAMS, 1 DJEMBE
PICNIC_BASKET2 = "PICNIC_BASKET2"  # 4 CROISSANTS, 2 JAMS
CROISSANTS = "CROISSANTS"
JAMS = "JAMS"
DJEMBE = "DJEMBE"

LIMIT_BSK1 = 60
LIMIT_BSK2 = 100
LIMIT_COMPONENT = 60  # component position cap (adjust based on your risk profile)

THRESHOLD_HIGH = 1.0
THRESHOLD_LOW = -1.0
PRICE_OFFSET = 1.0  # nudge to improve fill rate

class Trader:
    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        return (min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys())) / 2

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}
        prices = {}
        orders = {}

        for product, order_depth in state.order_depths.items():
            mid = self.get_mid_price(order_depth)
            if mid is not None:
                prices[product] = mid

        pos = {p: state.position.get(p, 0) for p in [
            PICNIC_BASKET1, PICNIC_BASKET2, CROISSANTS, JAMS, DJEMBE
        ]}

        # ------ BASKET 1 STRATEGY ------
        if all(p in prices for p in [PICNIC_BASKET1, CROISSANTS, JAMS, DJEMBE]):
            pb1_price = prices[PICNIC_BASKET1]
            fair_value = 6 * prices[CROISSANTS] + 3 * prices[JAMS] + prices[DJEMBE]
            spread = pb1_price - fair_value
            orders[PICNIC_BASKET1] = []

            if spread > THRESHOLD_HIGH and pos[PICNIC_BASKET1] > -LIMIT_BSK1:
                # Basket is overpriced → sell basket, buy components
                qty = min(10, LIMIT_BSK1 + pos[PICNIC_BASKET1])
                orders[PICNIC_BASKET1].append(Order(PICNIC_BASKET1, int(fair_value - PRICE_OFFSET), -qty))
                print(f"SELL PB1: {qty} @ {fair_value - PRICE_OFFSET} (spread {spread:.2f})")

                # Buy components
                orders.setdefault(CROISSANTS, []).append(Order(CROISSANTS, int(prices[CROISSANTS] + PRICE_OFFSET), 6 * qty))
                orders.setdefault(JAMS, []).append(Order(JAMS, int(prices[JAMS] + PRICE_OFFSET), 3 * qty))
                orders.setdefault(DJEMBE, []).append(Order(DJEMBE, int(prices[DJEMBE] + PRICE_OFFSET), qty))
                print(f"BUY components for PB1")

            elif spread < THRESHOLD_LOW and pos[PICNIC_BASKET1] < LIMIT_BSK1:
                # Basket underpriced → buy basket, sell components
                qty = min(10, LIMIT_BSK1 - pos[PICNIC_BASKET1])
                orders[PICNIC_BASKET1].append(Order(PICNIC_BASKET1, int(fair_value + PRICE_OFFSET), qty))
                print(f"BUY PB1: {qty} @ {fair_value + PRICE_OFFSET} (spread {spread:.2f})")

                # Sell components
                orders.setdefault(CROISSANTS, []).append(Order(CROISSANTS, int(prices[CROISSANTS] - PRICE_OFFSET), -6 * qty))
                orders.setdefault(JAMS, []).append(Order(JAMS, int(prices[JAMS] - PRICE_OFFSET), -3 * qty))
                orders.setdefault(DJEMBE, []).append(Order(DJEMBE, int(prices[DJEMBE] - PRICE_OFFSET), -qty))
                print(f"SELL components for PB1")

        # ------ BASKET 2 STRATEGY ------
        if all(p in prices for p in [PICNIC_BASKET2, CROISSANTS, JAMS]):
            pb2_price = prices[PICNIC_BASKET2]
            fair_value = 4 * prices[CROISSANTS] + 2 * prices[JAMS]
            spread = pb2_price - fair_value
            orders[PICNIC_BASKET2] = []

            if spread > THRESHOLD_HIGH and pos[PICNIC_BASKET2] > -LIMIT_BSK2:
                # Overpriced basket → sell basket, buy components
                qty = min(10, LIMIT_BSK2 + pos[PICNIC_BASKET2])
                orders[PICNIC_BASKET2].append(Order(PICNIC_BASKET2, int(fair_value - PRICE_OFFSET), -qty))
                print(f"SELL PB2: {qty} @ {fair_value - PRICE_OFFSET} (spread {spread:.2f})")

                orders.setdefault(CROISSANTS, []).append(Order(CROISSANTS, int(prices[CROISSANTS] + PRICE_OFFSET), 4 * qty))
                orders.setdefault(JAMS, []).append(Order(JAMS, int(prices[JAMS] + PRICE_OFFSET), 2 * qty))
                print(f"BUY components for PB2")

            elif spread < THRESHOLD_LOW and pos[PICNIC_BASKET2] < LIMIT_BSK2:
                # Underpriced basket → buy basket, sell components
                qty = min(10, LIMIT_BSK2 - pos[PICNIC_BASKET2])
                orders[PICNIC_BASKET2].append(Order(PICNIC_BASKET2, int(fair_value + PRICE_OFFSET), qty))
                print(f"BUY PB2: {qty} @ {fair_value + PRICE_OFFSET} (spread {spread:.2f})")

                orders.setdefault(CROISSANTS, []).append(Order(CROISSANTS, int(prices[CROISSANTS] - PRICE_OFFSET), -4 * qty))
                orders.setdefault(JAMS, []).append(Order(JAMS, int(prices[JAMS] - PRICE_OFFSET), -2 * qty))
                print(f"SELL components for PB2")

        return orders, 1, "BASKET_ARBITRAGE_WITH_COMPONENTS"
