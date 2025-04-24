from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math

strikes = [9500, 9750, 10000, 10250, 10500]
# strikes = [10250]

class Product:
    ROCK_VOUCHERS = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in strikes]
    ROCK = "VOLCANIC_ROCK"


PARAMS = {
    "VOLCANIC_ROCK_VOUCHER_9500": {
        "mean_volatility": 0.1959997370608378,
        "strike": 9500,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 30,
        "zscore_threshold": 2.1
    },
    "VOLCANIC_ROCK_VOUCHER_9750": {
        "mean_volatility": 0.1959997370608378,
        "strike": 9750,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 30,
        "zscore_threshold": 2.1
    },
    "VOLCANIC_ROCK_VOUCHER_10000": {
        "mean_volatility": 0.1959997370608378,
        "strike": 10000,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 30,
        "zscore_threshold": 2.1
    },
    "VOLCANIC_ROCK_VOUCHER_10250": {
        "mean_volatility": 0.1959997370608378,
        "strike": 10250,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 30,
        "zscore_threshold": 2.1
    },
    "VOLCANIC_ROCK_VOUCHER_10500": {
        "mean_volatility": 0.1959997370608378,
        "strike": 10500,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 30,
        "zscore_threshold": 2.1
    },
}

from math import log, sqrt, exp
from statistics import NormalDist

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {

            Product.ROCK: 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }


    def get_rock_voucher_mid_price(
        self, rock_voucher_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(rock_voucher_order_depth.buy_orders) > 0
            and len(rock_voucher_order_depth.sell_orders) > 0
        ):
            best_bid = max(rock_voucher_order_depth.buy_orders.keys())
            best_ask = min(rock_voucher_order_depth.sell_orders.keys())
            traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_coupon_price"]
        
    def rock_hedge_orders(
        self,
        rock_order_depth: OrderDepth,
        rock_voucher_order_depth: OrderDepth,
        rock_voucher_orders: List[Order],
        rock_position: int,
        rock_voucher_position: int,
        delta: float
    ) -> List[Order]:
        if rock_voucher_orders == None or len(rock_voucher_orders) == 0:
            rock_voucher_position_after_trade = rock_voucher_position
        else:
            rock_voucher_position_after_trade = rock_voucher_position + sum(order.quantity for order in rock_voucher_orders)
        
        target_rock_position = -delta * rock_voucher_position_after_trade
        
        if target_rock_position == rock_position:
            return None
        
        target_rock_quantity = target_rock_position - rock_position

        orders: List[Order] = []
        if target_rock_quantity > 0:
            # Buy ROCK
            best_ask = min(rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_rock_quantity),
                self.LIMIT[Product.ROCK] - rock_position,
            )
            if quantity > 0:
                orders.append(Order(Product.ROCK, best_ask, round(quantity)))
        
        elif target_rock_quantity < 0:
            # Sell ROCK
            best_bid = max(rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_rock_quantity),
                self.LIMIT[Product.ROCK] + rock_position,
            )
            if quantity > 0:
                orders.append(Order(Product.ROCK, best_bid, -round(quantity)))
        
        return orders

    def rock_voucher_orders(
        self,
        voucher,
        rock_voucher_order_depth: OrderDepth,
        rock_voucher_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> List[Order]:
        traderData['past_coupon_vol'].append(volatility)
        if len(traderData['past_coupon_vol']) < self.params[voucher]['std_window']:
            return None, None

        if len(traderData['past_coupon_vol']) > self.params[voucher]['std_window']:
            traderData['past_coupon_vol'].pop(0)
        
        # vol_z_score = (volatility - self.params[voucher]['mean_volatility']) / np.std(traderData['past_coupon_vol'])
        vol_std = np.std(traderData['past_coupon_vol'])
        if vol_std < 1e-6:
            vol_z_score = 0  # Treat as stable
        else:
            vol_z_score = (volatility - self.params[voucher]['mean_volatility']) / vol_std

        if (
            vol_z_score 
            >= self.params[voucher]['zscore_threshold']
        ):
            if rock_voucher_position != -self.LIMIT[voucher]:
                target_rock_voucher_position = -self.LIMIT[voucher]
                if len(rock_voucher_order_depth.buy_orders) > 0:
                    best_bid = max(rock_voucher_order_depth.buy_orders.keys())
                    target_quantity = abs(target_rock_voucher_position - rock_voucher_position)
                    quantity = min(
                        target_quantity,
                        abs(rock_voucher_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(voucher, best_bid, -quantity)], []
                    else:
                        return [Order(voucher, best_bid, -quantity)], [Order(voucher, best_bid, -quote_quantity)]

        elif (
            vol_z_score
            <= -self.params[voucher]["zscore_threshold"]
        ):
            if rock_voucher_position != self.LIMIT[voucher]:
                target_rock_voucher_position = self.LIMIT[voucher]
                if len(rock_voucher_order_depth.sell_orders) > 0:
                    best_ask = min(rock_voucher_order_depth.sell_orders.keys())
                    target_quantity = abs(target_rock_voucher_position - rock_voucher_position)
                    quantity = min(
                        target_quantity,
                        abs(rock_voucher_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(voucher, best_ask, quantity)], []
                    else:
                        return [Order(voucher, best_ask, quantity)], [Order(voucher, best_ask, quote_quantity)]

        return None, None



    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        day = (state.timestamp) / 1000000 / 250
        T = max(1e-6, (7 - day) / 7)

        # result = self.trading_strategy(state)

        for voucher in Product.ROCK_VOUCHERS:
            if voucher not in traderObject:
                traderObject[voucher] = {
                    "prev_coupon_price": 0,
                    "past_coupon_vol": []
                }

            if (
                voucher in self.params
                and voucher in state.order_depths
            ):
                rock_voucher_position = (
                    state.position[voucher]
                    if voucher in state.position
                    else 0
                )

                rock_position = (
                    state.position[Product.ROCK]
                    if Product.ROCK in state.position
                    else 0
                )

                rock_order_depth = state.order_depths[Product.ROCK]
                rock_voucher_order_depth = state.order_depths[voucher]
                rock_mid_price = (
                    min(rock_order_depth.buy_orders.keys())
                    + max(rock_order_depth.sell_orders.keys())
                ) / 2
                rock_voucher_mid_price = self.get_rock_voucher_mid_price(
                    rock_voucher_order_depth, traderObject[voucher]
                )
                tte = (
                    T
            
                )
                volatility = BlackScholes.implied_volatility(
                    rock_voucher_mid_price,
                    rock_mid_price,
                    self.params[voucher]["strike"],
                    tte,
                )
                delta = BlackScholes.delta(
                    rock_mid_price,
                    self.params[voucher]["strike"],
                    tte,
                    volatility,
                )
        
                rock_voucher_take_orders, rock_voucher_make_orders = self.rock_voucher_orders(
                    voucher,
                    state.order_depths[voucher],
                    rock_voucher_position,
                    traderObject[voucher],
                    volatility,
                )

                rock_orders = self.rock_hedge_orders(
                    state.order_depths[Product.ROCK],
                    state.order_depths[voucher],
                    rock_voucher_take_orders,
                    rock_position,
                    rock_voucher_position,
                    delta
                )

                if rock_voucher_take_orders != None or rock_voucher_make_orders != None:
                    result[voucher] = rock_voucher_take_orders + rock_voucher_make_orders

                if rock_orders is not None:
                    if Product.ROCK not in result:
                      result[Product.ROCK] = rock_orders
                    else:
                      result[Product.ROCK].extend(rock_orders)


        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData