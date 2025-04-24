from typing import Dict, List
import numpy as np
import math
import statistics
import jsonpickle
from datamodel import Order, OrderDepth, TradingState

class Trader:
    def __init__(self):
        self.voucher_pnl = 0.0
        self.strikes = [9500, 9750, 10000, 10250, 10500]
        self.voucher_products = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in self.strikes]
        self.basic_products = ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]
        self.rock_product = "VOLCANIC_ROCK"
        self.window_size = 5
        self.limit = 200
        self.basic_limits = {p: 50 for p in self.basic_products}
        # Boost SQUID_INK exposure dynamically on Day 0
        self.default_squid_limit = 100
        self.day0_squid_limit = 200  # Stronger allocation on Day 0
        self.tick_count = 0
        self.price_history = {p: [] for p in self.basic_products}
        self.active_trades = {p: None for p in self.basic_products}
        self.rock_prices = []  # for adaptive volatility estimation

    def norm_cdf(self, x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def bs_call_price(self, S, K, T, sigma, r=0):
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)

    def implied_vol(self, S, K, T, market_price, r=0):
        low, high = 1e-6, 3.0
        for _ in range(100):
            mid = (low + high) / 2
            price = self.bs_call_price(S, K, T, mid, r)
            if abs(price - market_price) < 1e-6:
                return mid
            if price > market_price:
                high = mid
            else:
                low = mid
        return mid

    def run(self, state: TradingState):
        self.tick_count += 1
        result = {}
        conversions = 0

        mid_prices = {}
        for product, depth in state.order_depths.items():
            if depth.buy_orders and depth.sell_orders:
                best_bid = max(depth.buy_orders.keys())
                best_ask = min(depth.sell_orders.keys())
                mid_prices[product] = (best_bid + best_ask) / 2

        rock_price = mid_prices.get(self.rock_product)
        if rock_price is not None:
            self.rock_prices.append(rock_price)
            if len(self.rock_prices) > 20:
                self.rock_prices.pop(0)
        rock_vol = np.std(self.rock_prices) if len(self.rock_prices) >= 5 else 1

        def dynamic_std_multiplier(vol):
            return max(1.0, min(2.0, 1.2 + 5 * vol / 100))

        def dynamic_stop_loss_multiplier(vol):
            return max(1.5, min(3.5, 2.0 + 5 * vol / 100))

        def dynamic_max_hold(vol):
            return int(max(10, min(40, 30 - 100 * vol / 100)))

        day = state.timestamp // 10000

        for product in self.basic_products:
            if product not in state.order_depths:
                continue
            order_depth = state.order_depths[product]
            current_price = mid_prices.get(product)
            if current_price is None:
                continue

            self.price_history[product].append(current_price)
            if len(self.price_history[product]) > self.window_size:
                self.price_history[product].pop(0)

            if len(self.price_history[product]) < 2:
                continue

            sma = sum(self.price_history[product]) / len(self.price_history[product])
            std = statistics.stdev(self.price_history[product])
            std_mult = dynamic_std_multiplier(std)
            stop_mult = dynamic_stop_loss_multiplier(std)
            max_hold = dynamic_max_hold(std)
            if std > 10:
                stop_mult *= 0.8
                max_hold = max(10, max_hold - 10)

            if day == 0:
                std_mult *= 0.9
                stop_mult *= 1.1
                max_hold += 5

            upper_band = sma + std_mult * std
            lower_band = sma - std_mult * std
            position = state.position.get(product, 0)
            if product == "SQUID_INK" and day == 0:
                recent_prices = self.price_history[product][-self.window_size:]
                squid_vol = statistics.stdev(recent_prices) if len(recent_prices) >= 2 else 1
                avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else 100
                book_depth = sum(abs(v) for v in order_depth.buy_orders.values()) + sum(abs(v) for v in order_depth.sell_orders.values())
                liquidity_boost = 1 + book_depth / 200  # normalize by typical depth
                alpha = 0.7  # increased sensitivity to volatility
                limit = int(self.default_squid_limit * (1 + alpha * squid_vol / avg_price) * liquidity_boost)
                if squid_vol < 5:
                    limit = min(limit, 400)  # Increase if calm
                else:
                    limit = min(limit, 300)  # Default cap

                # Additional signal: reversion strength factor
                deviation = abs(current_price - avg_price)
                reversion_signal = deviation / (squid_vol + 1e-6)  # avoid div by zero
                if reversion_signal > 1.5:
                    limit = int(limit * 1.2)  # if swing is strong, expect reversion
            elif (day == 0 or day == 2) and product == "SQUID_INK":
                recent_prices = self.price_history[product][-self.window_size:]
                squid_vol = statistics.stdev(recent_prices) if len(recent_prices) >= 2 else 1
                avg_price = sum(recent_prices) / len(recent_prices) if recent_prices else 100
                book_depth = sum(abs(v) for v in order_depth.buy_orders.values()) + sum(abs(v) for v in order_depth.sell_orders.values())
                liquidity_boost = 1 + book_depth / 200
                alpha = 0.5
                limit = int(self.default_squid_limit * (1 + alpha * squid_vol / avg_price) * liquidity_boost)
                if squid_vol < 5:
                    limit = min(limit, 400)
                else:
                    limit = min(limit, 300)
                deviation = abs(current_price - avg_price)
                reversion_signal = deviation / (squid_vol + 1e-6)
                if reversion_signal > 2:
                    limit = int(limit * 1.2)
            else:
                limit = self.basic_limits[product]
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            orders = []

            if self.active_trades[product] is None:
                if current_price < lower_band and best_ask:
                    qty = min(limit - position, order_depth.sell_orders[best_ask])
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                        self.active_trades[product] = {"position": "long", "entry_tick": self.tick_count, "target_price": sma, "stop_loss": current_price - stop_mult * std, "max_hold": max_hold}
                elif current_price > upper_band and best_bid:
                    qty = min(position + limit, order_depth.buy_orders[best_bid])
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                        self.active_trades[product] = {"position": "short", "entry_tick": self.tick_count, "target_price": sma, "stop_loss": current_price + stop_mult * std, "max_hold": max_hold}
            else:
                trade = self.active_trades[product]
                time_in_trade = self.tick_count - trade["entry_tick"]
                if trade["position"] == "long":
                    if current_price >= trade["target_price"] or current_price <= trade["stop_loss"] or time_in_trade >= trade["max_hold"]:
                        if best_bid:
                            qty = min(position, order_depth.buy_orders[best_bid])
                            if qty > 0:
                                orders.append(Order(product, best_bid, -qty))
                                self.active_trades[product] = None
                elif trade["position"] == "short":
                    if current_price <= trade["target_price"] or current_price >= trade["stop_loss"] or time_in_trade >= trade["max_hold"]:
                        if best_ask:
                            qty = min(limit + position, order_depth.sell_orders[best_ask])
                            if qty > 0:
                                orders.append(Order(product, best_ask, qty))
                                self.active_trades[product] = None

            if orders:
                result[product] = orders

        if self.rock_product in mid_prices:
            spot = mid_prices[self.rock_product]
            T = max(1e-6, (7 - day) / 7)

            for product in self.voucher_products:
                if day == 0:
                    continue  # fully disable voucher trading on Day 0

                if product in state.order_depths and product in mid_prices:
                    strike = int(product.split('_')[-1])

                    if day == 0 and abs(strike - spot) > 250:
                        continue  # avoid risky deep OTM vouchers early

                    market_price = mid_prices[product]
                    iv = self.implied_vol(spot, strike, T, market_price)
                    fair_price = self.bs_call_price(spot, strike, T, iv)
                    mispricing = market_price - fair_price
                    order_depth = state.order_depths[product]
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                    orders = []

                    vol_multiplier = min(max(iv, 0.1), 1.0)
                    mispricing_thresh = 5 + 10 * vol_multiplier
                    qty_scale_factor = 15 / vol_multiplier
                    max_loss_per_voucher = 5000

                    if mispricing > mispricing_thresh and best_bid:
                        qty = int(self.limit * min(1.0, mispricing / qty_scale_factor))
                        if qty * best_bid > max_loss_per_voucher:
                            qty = max_loss_per_voucher // best_bid
                        qty = min(qty, order_depth.buy_orders[best_bid])
                        orders.append(Order(product, best_bid, -qty))
                        self.voucher_pnl += qty * best_bid
                    elif mispricing < -mispricing_thresh and best_ask:
                        qty = int(self.limit * min(1.0, abs(mispricing) / qty_scale_factor))
                        if qty * best_ask > max_loss_per_voucher:
                            qty = max_loss_per_voucher // best_ask
                        qty = min(qty, order_depth.sell_orders[best_ask])
                        orders.append(Order(product, best_ask, qty))
                        self.voucher_pnl -= qty * best_ask

                    if orders:
                        result[product] = orders

        trader_data = jsonpickle.encode({
            "price_history": self.price_history,
            "active_trades": self.active_trades,
            "tick_count": self.tick_count
        })

        if day == 0 and self.tick_count % 1000 == 0:
            print(f"[Day 0 Tick {self.tick_count}] PnL Snapshot: SQUID_INK={state.position.get('SQUID_INK', 0)}, RAINFOREST_RESIN={state.position.get('RAINFOREST_RESIN', 0)}, KELP={state.position.get('KELP', 0)}")
        print(f"[Tick {self.tick_count}] Voucher Arbitrage PnL: {self.voucher_pnl:.2f}")
        return result, conversions, trader_data

