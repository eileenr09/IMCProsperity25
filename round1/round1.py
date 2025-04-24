from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import numpy as np
import math

POSITION_LIMIT = 50

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

class Trader:
    def __init__(self):
        self.pnl = 0

    def execute(self, state, product, fair_price):
        FAIR_PRICE = fair_price
        # Retrieve the Order Depth containing all the market BUY and SELL orders
        # Products -> instances of the OrderDepth class
        order_depth: OrderDepth = state.order_depths[product]
        orders: list[Order] = []  
        current_pos = state.position.get(product, 0)
        edge = 3

        for ask_price, ask_amt in sorted(order_depth.sell_orders.items()):
            if ask_price < FAIR_PRICE:
                buy_volume = min(-ask_amt, POSITION_LIMIT - current_pos)
                if buy_volume > 0:
                    current_pos += buy_volume
                    print(f"BUY {buy_volume} at {ask_price}")
                    orders.append(Order(product, ask_price, buy_volume))

        for bid_price, bid_amt in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > FAIR_PRICE:
                sell_volume = min(bid_amt, current_pos + POSITION_LIMIT)
                if sell_volume > 0:
                    current_pos -= sell_volume
                    print(f"SELL {sell_volume} at {bid_price}")
                    orders.append(Order(product, bid_price, -sell_volume))
        if product == Product.RAINFOREST_RESIN:
        
          bid_price = FAIR_PRICE - edge
          ask_price = FAIR_PRICE + edge

          bid_volume = min(5, POSITION_LIMIT - current_pos)
          ask_volume = min(5, current_pos + POSITION_LIMIT)

          if bid_volume > 0:
              print(f"MAKE! BID {bid_volume} at {bid_price}")
              orders.append(Order(product, bid_price, bid_volume))

          if ask_volume > 0:
              print(f"MAKE! ASK {ask_volume} at {ask_price}")
              orders.append(Order(product, ask_price, -ask_volume))
          
        return orders
    
    def calculate_moving_average(self, prices: List[float], window: int) -> float:
        if len(prices) == 0:
            return 0
        if len(prices) < window:
            return sum(prices) / len(prices)  
        return sum(prices[-window:]) / window
    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:

        traderData = "SAMPLE" 
        conversions = 1 
        result = {}
        for product in state.order_depths.keys():
            order_depth = state.order_depths[product]
            orders = []

            if product == Product.RAINFOREST_RESIN:
                orders = self.execute(state, product, 10000)
                result[product] = orders

            elif product == Product.KELP or product == Product.SQUID_INK:
                historical_prices = []
                if product in state.market_trades:
                    trades = state.market_trades.get(product, [])
                    historical_prices = [trade.price for trade in trades]

                short_term_ma = self.calculate_moving_average(historical_prices, 5)
                long_term_ma = self.calculate_moving_average(historical_prices, 20)

                print(f"Short-Term MA: {short_term_ma}, Long-Term MA: {long_term_ma}")

                current_pos = state.position.get(product, 0)

                threshold = 0
                ma_diff = short_term_ma - long_term_ma

                if ma_diff > threshold:
                    for ask_price, ask_amt in sorted(state.order_depths[product].sell_orders.items()):
                        buy_volume = min(-ask_amt, POSITION_LIMIT - current_pos)
                        if buy_volume > 0:
                            current_pos += buy_volume
                            print(f"BUY {buy_volume} at {ask_price}")
                            orders.append(Order(product, ask_price, buy_volume))

                if ma_diff < -threshold:
                    for bid_price, bid_amt in sorted(state.order_depths[product].buy_orders.items(), reverse=True):
                        sell_volume = min(bid_amt, current_pos + POSITION_LIMIT)
                        if sell_volume > 0:
                            current_pos -= sell_volume
                            print(f"SELL {sell_volume} at {bid_price}")
                            orders.append(Order(product, bid_price, -sell_volume))

                result[product] = orders

            elif product == Product.SQUID_INK:
                historical_prices = []
                if product in state.market_trades:
                    trades = state.market_trades.get(product, [])
                    historical_prices = [trade.price for trade in trades]

                rolling_window = 10
                rolling_mean = self.calculate_moving_average(historical_prices, rolling_window)
                rolling_std = self.calculate_rolling_std(historical_prices, rolling_window)

                if rolling_std != 0: 
                    z_score = (historical_prices[-1] - rolling_mean) / rolling_std
                else:
                    z_score = 0  

                print(f"z-score: {z_score}, Rolling Mean: {rolling_mean}, Rolling Std Dev: {rolling_std}")

                buy_threshold = 0
                sell_threshold = 0

                if z_score < buy_threshold:
                    print(f"Buy Signal detected: z-score < {buy_threshold}")
                    for ask_price, ask_amt in sorted(order_depth.sell_orders.items()):
                        buy_volume = min(-ask_amt, POSITION_LIMIT - current_pos)
                        buy_volume = max(buy_volume, int(POSITION_LIMIT * 0.8))  
                        if buy_volume > 0:
                            current_pos += buy_volume
                            print(f"BUY {buy_volume} at {ask_price}")
                            orders.append(Order(Product.SQUID_INK, ask_price, buy_volume))

                elif z_score > sell_threshold:
                    print(f"Sell Signal detected: z-score > {sell_threshold}")
                    for bid_price, bid_amt in sorted(order_depth.buy_orders.items(), reverse=True):
                        sell_volume = min(bid_amt, current_pos + POSITION_LIMIT)
                        sell_volume = max(sell_volume, int(POSITION_LIMIT * 0.8)) 
                        if sell_volume > 0:
                            current_pos -= sell_volume
                            print(f"SELL {sell_volume} at {bid_price}")
                            orders.append(Order(Product.SQUID_INK, bid_price, -sell_volume))
                result[product] = orders

        if state.timestamp == 99900:
            for product, trades in state.own_trades.items():
              for trade in trades:
                if trade.buyer == "SUBMISSION":
                    self.pnl -= trade.price * trade.quantity
                    print(f"Buy {product} amt: {trade.quantity}; price: {trade.price}")
                if trade.seller == "SUBMISSION":
                    self.pnl += trade.price * trade.quantity
                    print(f"Sell {product} amt: {trade.quantity}; price: {trade.price}")

            print(f"Total PNL: {self.pnl}")

        return result, conversions, traderData



            


