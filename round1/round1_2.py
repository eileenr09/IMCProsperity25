from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math, statistics


class Product:
    RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID = "SQUID_INK"


PARAMS = {
    Product.RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.SQUID: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "soft_position_limit": 10,
    },
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.RESIN: 50, Product.KELP: 50, Product.SQUID: 50}
        self.historical_trades = []

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) :
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    # def starfruit_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
    #     if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_bid = max(order_depth.buy_orders.keys())
    #         filtered_ask = [
    #             price
    #             for price in order_depth.sell_orders.keys()
    #             if abs(order_depth.sell_orders[price])
    #             >= self.params[Product.STARFRUIT]["adverse_volume"]
    #         ]
    #         filtered_bid = [
    #             price
    #             for price in order_depth.buy_orders.keys()
    #             if abs(order_depth.buy_orders[price])
    #             >= self.params[Product.STARFRUIT]["adverse_volume"]
    #         ]
    #         mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
    #         mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
    #         if mm_ask == None or mm_bid == None:
    #             if traderObject.get("starfruit_last_price", None) == None:
    #                 mmmid_price = (best_ask + best_bid) / 2
    #             else:
    #                 mmmid_price = traderObject["starfruit_last_price"]
    #         else:
    #             mmmid_price = (mm_ask + mm_bid) / 2

    #         if traderObject.get("starfruit_last_price", None) != None:
    #             last_price = traderObject["starfruit_last_price"]
    #             last_returns = (mmmid_price - last_price) / last_price
    #             pred_returns = (
    #                 last_returns * self.params[Product.STARFRUIT]["reversion_beta"]
    #             )
    #             fair = mmmid_price + (mmmid_price * pred_returns)
    #         else:
    #             fair = mmmid_price
    #         traderObject["starfruit_last_price"] = mmmid_price
    #         return fair
    #     return None

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
        
    def calculate_moving_average(self, prices: List[float], window: int) -> float:
        if len(prices) == 0:
            return 0
        if len(prices) < window:
            return sum(prices) / len(prices)  
        return sum(prices[-window:]) / window
    
    def calculate_rolling_std(self, prices: List[float], window: int) -> float:
      if len(prices) == 0:
        return 0
      if len(prices) < window:
          return 0
      return statistics.stdev(prices[-window:])
      
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RESIN in self.params and Product.RESIN in state.order_depths:
            resin_position = (
                state.position[Product.RESIN]
                if Product.RESIN in state.position
                else 0
            )
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RESIN,
                    state.order_depths[Product.RESIN],
                    self.params[Product.RESIN]["fair_value"],
                    self.params[Product.RESIN]["take_width"],
                    resin_position,
                )
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RESIN,
                    state.order_depths[Product.RESIN],
                    self.params[Product.RESIN]["fair_value"],
                    self.params[Product.RESIN]["clear_width"],
                    resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RESIN,
                state.order_depths[Product.RESIN],
                self.params[Product.RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RESIN]["disregard_edge"],
                self.params[Product.RESIN]["join_edge"],
                self.params[Product.RESIN]["default_edge"],
                True,
                self.params[Product.RESIN]["soft_position_limit"],
            )
            result[Product.RESIN] = (
                resin_take_orders + resin_clear_orders + resin_make_orders
            )

        if Product.SQUID in state.order_depths:
            current_pos = state.position.get(Product.SQUID, 0)
            order_depth = state.order_depths[Product.SQUID]
            orders = []
            fair_value = 2000
            if Product.SQUID in state.market_trades:
                trades = state.market_trades[Product.SQUID]
                self.historical_trades.extend(trades)
                prices = [trade.price for trade in trades]
                
                # Calculate the moving average and rolling standard deviation
                window = 10  # You can modify the window size as needed
                moving_avg = self.calculate_moving_average(prices, window)
                rolling_std = self.calculate_rolling_std(prices, window)
                
                # Decision based on moving average and standard deviation
                fair_value = moving_avg
                take_width = rolling_std * 2
                upper = moving_avg + take_width
                lower = moving_avg - take_width
                if prices:
                  last = prices[-1]
                  
                  if last < lower and rolling_std < 1:  # Example condition for taking buy orders
                      buy_orders, buy_order_volume, sell_order_volume = self.take_orders(
                          Product.SQUID,
                          order_depth,
                          fair_value,
                          take_width,
                          current_pos,
                      )
                      orders.extend(buy_orders)

                  if last > upper and rolling_std < 1:  # Example condition for taking sell orders
                      sell_orders, buy_order_volume, sell_order_volume = self.take_orders(
                          Product.SQUID,
                          order_depth,
                          fair_value,
                          take_width,
                          current_pos,
                      )
                      orders.extend(sell_orders)

            # Making orders based on position and market conditions
            squid_make_orders, buy_order_volume, sell_order_volume = self.make_orders(
                Product.SQUID,
                order_depth,
                fair_value,
                current_pos,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID]["disregard_edge"],
                self.params[Product.SQUID]["join_edge"],
                self.params[Product.SQUID]["default_edge"],
                True,
                self.params[Product.SQUID]["soft_position_limit"],
            )
            orders.extend(squid_make_orders)

            result[Product.SQUID] = orders
        
        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData