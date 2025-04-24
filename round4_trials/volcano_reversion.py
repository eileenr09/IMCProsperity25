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
        # self.strikes = [10000, 10250, 10500]
        self.voucher_products = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in self.strikes]
        self.rock_product = "VOLCANIC_ROCK"
        self.window_size_vol = 5
        self.limit_vol = 200
        self.active_positions = {}  # { symbol: { direction, entry_price, qty, entry_tick } }
        self.tick_count_vol = 0
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
        self.tick_count_vol += 1
        result = {}
        conversions = 0

        mid_prices = {}
        best_asks = {}
        best_bids = {}
        for product, order_depth in state.order_depths.items():
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                best_asks[product] = best_ask
                best_bids[product] = best_bid
                mid_prices[product] = (best_bid + best_ask) / 2

        rock_price = mid_prices.get(self.rock_product)
        if rock_price is not None:
            self.rock_prices.append(rock_price)
            if len(self.rock_prices) > 20:
                self.rock_prices.pop(0)

        day = state.timestamp // 10000

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
                    
                    orders = []

                    vol_multiplier = min(max(iv, 0.1), 1.0)
                    mispricing_thresh = 5 + 10 * vol_multiplier
                    qty_scale_factor = 15 / vol_multiplier
                    max_loss_per_voucher = 5000

                    if mispricing > mispricing_thresh and best_bids[product]:
                        qty = int(self.limit_vol * min(1.0, mispricing / qty_scale_factor))
                        if qty * best_bids[product] > max_loss_per_voucher:
                            qty = max_loss_per_voucher // best_bids[product]
                        qty = min(qty, order_depth.buy_orders[best_bids[product]])
                        orders.append(Order(product, best_bids[product], -qty))
                        self.voucher_pnl += qty * best_bids[product]
                    elif mispricing < -mispricing_thresh and best_asks[product]:
                        qty = int(self.limit_vol * min(1.0, abs(mispricing) / qty_scale_factor))
                        if qty * best_asks[product] > max_loss_per_voucher:
                            qty = max_loss_per_voucher // best_asks[product]
                        qty = min(qty, order_depth.sell_orders[best_asks[product]])
                        orders.append(Order(product, best_asks[product], qty))
                        self.voucher_pnl -= qty * best_asks[product]
                    close_threshold = 1.5  # Small band where we consider price to have reverted
                    symbol = product

                    # If position is open and mispricing has reverted
                    pos_qty = state.position.get(symbol, 0)
                    if symbol in state.position and abs(mispricing) < close_threshold and pos_qty != 0:
                        # Decide which side to close
                        if pos_qty > 0 and best_bids.get(symbol):  # Long -> sell to close
                            close_qty = min(pos_qty, order_depth.buy_orders[best_bids[symbol]])
                            orders.append(Order(symbol, best_bids[symbol], -close_qty))
                            self.voucher_pnl += close_qty * best_bids[symbol]
                        elif pos_qty < 0 and best_asks.get(symbol):  # Short -> buy to close
                            close_qty = min(-pos_qty, order_depth.sell_orders[best_asks[symbol]])
                            orders.append(Order(symbol, best_asks[symbol], close_qty))
                            self.voucher_pnl -= close_qty * best_asks[symbol]
                    if orders:
                        result[product] = orders

        trader_data = ""

        if day == 0 and self.tick_count_vol % 1000 == 0:
            print(f"[Day 0 Tick {self.tick_count_vol}] PnL Snapshot: SQUID_INK={state.position.get('SQUID_INK', 0)}, RAINFOREST_RESIN={state.position.get('RAINFOREST_RESIN', 0)}, KELP={state.position.get('KELP', 0)}")
        print(f"[Tick {self.tick_count_vol}] Voucher Arbitrage PnL: {self.voucher_pnl:.2f}")
        return result, conversions, trader_data

