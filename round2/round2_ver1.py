from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


PICNIC_BASKET1 = "PICNIC_BASKET1" # 6 CROISSANTS, 3 JAMS, 1 DJEMBE
PICNIC_BASKET2 = "PICNIC_BASKET2" # 4 CROISSANTS, 2 JAMS
CROISSANTS = "CROISSANTS"
JAMS = "JAMS"
DJEMBE = "DJEMBE"
products = [CROISSANTS, JAMS, DJEMBE]

LIMIT_BSK1 = 60
LIMIT_BSK2 = 100
a, b, c = 0.2, -0.2, 2

class Trader:
    def get_mid_price(self, order_depth: OrderDepth) -> float:
      if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
        return None
      best_ask = min(order_depth.sell_orders.keys())
      best_bid = max(order_depth.buy_orders.keys())
      return (best_ask + best_bid) / 2

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        just fix three numbers a, b, c. 
        if the difference between etf - the components > a, 
        then sell the etf at fair - c. 
        if it is < b, then buy the etf at fair + c 
        """
        # Initialize the method output dict as an empty dict

        result = {}
        
        # Iterate over all the keys (the available products) contained in the order depths
        product2price = dict()
        for product in state.order_depths.keys():
          order_depth: OrderDepth = state.order_depths[product]
          mid_price = self.get_mid_price(order_depth)
          product2price[product] = mid_price
        if PICNIC_BASKET1 in product2price.keys():
          if CROISSANTS in product2price.keys() and JAMS in product2price.keys() and DJEMBE in product2price.keys():
            orders_bsk1: list[Order] = []
            bsk1, c, j, d = product2price[PICNIC_BASKET1], product2price[CROISSANTS], product2price[JAMS], product2price[DJEMBE]
            fair_price = 6*c + 3*j + d
            if bsk1 - fair_price > a:
               if PICNIC_BASKET1 in state.position:
                  pos_bsk1 = state.position[PICNIC_BASKET1]
                  remaining_pos = pos_bsk1 + LIMIT_BSK1
                  sell_price = fair_price - c
                  print(f"SELL BASKET 1 at {sell_price} with amount {remaining_pos * 0.5}")
                  orders_bsk1.append(Order(PICNIC_BASKET1, int(sell_price), -remaining_pos * 0.5))
            if bsk1 - fair_price < b:
               if PICNIC_BASKET1 in state.position:
                  pos_bsk1 = state.position[PICNIC_BASKET1]
                  remaining_pos = LIMIT_BSK1 - pos_bsk1
                  bid_price = fair_price + c
                  print(f"BID BASKET 1 at {bid_price} with amount {remaining_pos * 0.5}")
                  orders_bsk1.append(Order(PICNIC_BASKET1, int(bid_price), remaining_pos * 0.5))
            result[PICNIC_BASKET1] = orders_bsk1

        if PICNIC_BASKET2 in product2price.keys():
          if CROISSANTS in product2price.keys() and JAMS in product2price.keys():
            orders_bsk2: list[Order] = []
            bsk2, c, j= product2price[PICNIC_BASKET2], product2price[CROISSANTS], product2price[JAMS]
            fair_price = 4*c + 2*j
            if bsk2 - fair_price > a:
               if PICNIC_BASKET2 in state.position:
                  pos_bsk2 = state.position[PICNIC_BASKET2]
                  remaining_pos = pos_bsk2 + LIMIT_BSK2
                  sell_price = fair_price - c
                  print(f"SELL BASKET 2 at {sell_price} with amount {remaining_pos * 0.5}")
                  orders_bsk2.append(Order(PICNIC_BASKET2, int(sell_price), -remaining_pos * 0.5))
            if bsk2 - fair_price < b:
               if PICNIC_BASKET2 in state.position:
                  pos_bsk2 = state.position[PICNIC_BASKET2]
                  remaining_pos = LIMIT_BSK2 - pos_bsk2
                  bid_price = fair_price + c
                  print(f"BID BASKET 2 at {bid_price} with amount {remaining_pos * 0.5}")
                  orders_bsk2.append(Order(PICNIC_BASKET2, int(bid_price), remaining_pos * 0.5))
            result[PICNIC_BASKET2] = orders_bsk2
    
        traderData = "SAMPLE"
        
        conversions = 1 
        
        return result, conversions, traderData
