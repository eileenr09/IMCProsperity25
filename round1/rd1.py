from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math, statistics
from collections import deque


class Product:
    RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID = "SQUID_INK"


PARAMS = {
    Product.RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 50,
    },
    Product.SQUID: {
        "take_width": 1,
        "clear_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "window_size": 20,  
        "std_dev_multiplier": 1.7,  
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0.5,
        "trend_window": 10,  
        "min_trend_strength": 0.5, 
    }
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.RESIN: 50, Product.KELP: 50, Product.SQUID: 50}
        self.position_limits = self.LIMIT  
        
        self.price_history = {
            Product.SQUID: deque(maxlen=self.params[Product.SQUID]["window_size"]),
            Product.KELP: deque(maxlen=self.params[Product.KELP]["trend_window"]),
        }

        self.squid_vwap = None
        self.kelp_vwap = None
        
        self.prev_positions = {}
        self.prev_prices = {}

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
    
    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate the current mid price from order book"""
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return None
            
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2

    def get_squid_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        """Calculate fair value for Squid Ink using mean-reversion strategy"""
        current_price = self.get_mid_price(order_depth)
        if current_price is None:
            return None
          
        self.price_history[Product.SQUID].append(current_price)
        if len(self.price_history[Product.SQUID]) == 0:
            return current_price
            
        sma = sum(self.price_history[Product.SQUID]) / len(self.price_history[Product.SQUID])
        
        # Calculate Bollinger Bands
        std_dev = statistics.stdev(self.price_history[Product.SQUID]) if len(self.price_history[Product.SQUID]) > 1 else 0
        upper_band = sma + self.params[Product.SQUID]["std_dev_multiplier"] * std_dev
        lower_band = sma - self.params[Product.SQUID]["std_dev_multiplier"] * std_dev
        
        # Determine if we should trade based on mean reversion
        if current_price > upper_band:
            # Price is too high - likely to revert down
            return lower_band  # Target lower band for selling
        elif current_price < lower_band:
            # Price is too low - likely to revert up
            return upper_band  # Target upper band for buying
        else:
            return sma  # Near fair value

    def get_kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        """Calculate fair value for Kelp using trend-following strategy"""
        current_price = self.get_mid_price(order_depth)
        if current_price is None:
            return None
            
        # Add current price to history
        self.price_history[Product.KELP].append(current_price)
        
        if len(self.price_history[Product.KELP]) < 2:
            return current_price
            
        # Calculate linear regression to determine trend
        x = np.arange(len(self.price_history[Product.KELP]))
        y = np.array(self.price_history[Product.KELP])
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Determine trend strength and direction
        if abs(slope) > self.params[Product.KELP]["min_trend_strength"]:
            if slope > 0:
                # Upward trend - fair value is higher than current price
                return current_price * (1 + slope)  # Project slightly ahead
            else:
                # Downward trend - fair value is lower than current price
                return current_price * (1 + slope)  # Project slightly lower
        else:
            # No strong trend - stay neutral
            return current_price

    def execute_trade(self, product: str, fair_value: float, state: TradingState) -> List[Order]:
        """Execute trades based on calculated fair value"""
        orders = []
        order_depth = state.order_depths.get(product, None)
        if order_depth is None:
            return orders
            
        current_position = state.position.get(product, 0)
        position_limit = self.LIMIT[product]
        
        # Get current best bid/ask
        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None
        
        if best_bid is None or best_ask is None:
            return orders
            
        current_price = (best_bid + best_ask) / 2
        
        if product == Product.SQUID:
            # Mean-reversion strategy
            if fair_value > current_price * 1.01:  # Price is below fair value
                # Buy opportunity
                buy_size = min(position_limit - current_position, position_limit)
                if buy_size > 0:
                    price = min(best_ask, fair_value - 1)  # Try to buy at or below fair value
                    orders.append(Order(product, price, buy_size))
            elif fair_value < current_price * 0.99:  # Price is above fair value
                # Sell opportunity
                sell_size = min(current_position + position_limit, position_limit)
                if sell_size > 0:
                    price = max(best_bid, fair_value + 1)  # Try to sell at or above fair value
                    orders.append(Order(product, price, -sell_size))
                    
        elif product == Product.KELP:
            # Trend-following strategy
            if fair_value > current_price * 1.01:  # Upward trend
                # Buy opportunity
                buy_size = min(position_limit - current_position, position_limit)
                if buy_size > 0:
                    price = min(best_ask, fair_value - 1)
                    orders.append(Order(product, price, buy_size))
            elif fair_value < current_price * 0.99:  # Downward trend
                # Sell opportunity
                sell_size = min(current_position + position_limit, position_limit)
                if sell_size > 0:
                    price = max(best_bid, fair_value + 1)
                    orders.append(Order(product, price, -sell_size))
                    
        return orders
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
    
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)
        
        result = {}
        
        for product in [Product.RESIN, Product.SQUID, Product.KELP]:
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                current_position = state.position.get(product, 0)
                
                if product == Product.RESIN:
                    resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                        Product.RESIN,
                        order_depth,
                        self.params[Product.RESIN]["fair_value"],
                        self.params[Product.RESIN]["take_width"],
                        current_position,
                    )
                    resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                        Product.RESIN,
                        order_depth,
                        self.params[Product.RESIN]["fair_value"],
                        self.params[Product.RESIN]["clear_width"],
                        current_position,
                        buy_order_volume,
                        sell_order_volume,
                    )
                    resin_make_orders, _, _ = self.make_orders(
                        Product.RESIN,
                        order_depth,
                        self.params[Product.RESIN]["fair_value"],
                        current_position,
                        buy_order_volume,
                        sell_order_volume,
                        self.params[Product.RESIN]["disregard_edge"],
                        self.params[Product.RESIN]["join_edge"],
                        self.params[Product.RESIN]["default_edge"],
                        True,
                        self.params[Product.RESIN]["soft_position_limit"],
                    )
                    result[Product.RESIN] = resin_take_orders + resin_clear_orders + resin_make_orders
                
                elif product == Product.SQUID:
                    fair_value = self.get_squid_fair_value(order_depth, traderObject)
                    if fair_value is not None:
                        orders = self.execute_trade(Product.SQUID, fair_value, state)
                        result[Product.SQUID] = orders
                
                elif product == Product.KELP:
                    fair_value = self.get_kelp_fair_value(order_depth, traderObject)
                    if fair_value is not None:
                        orders = self.execute_trade(Product.KELP, fair_value, state)
                        # result[Product.KELP] = orders
        
        traderData = {
            "price_history": {
                Product.SQUID: list(self.price_history[Product.SQUID]),
                Product.KELP: list(self.price_history[Product.KELP]),
            },
            "prev_positions": state.position,
            "prev_prices": {
                prod: (max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None,
                min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None
                ) for prod, order_depth in state.order_depths.items()
            }
        }
        
        conversions = 0
        return result, conversions, jsonpickle.encode(traderData)