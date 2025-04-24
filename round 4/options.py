from typing import Dict, List
import math
from statistics import NormalDist
from datamodel import OrderDepth, TradingState, Order

def BS_CALL(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black–Scholes call price."""
    N = NormalDist().cdf
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S * N(d1) - K * math.exp(-r*T) * N(d2)

class Trader:
    def __init__(self):
        # Underlying and voucher symbols
        self.rock_product     = "VOLCANIC_ROCK"
        self.strikes          = [9500, 9750, 10000, 10250, 10500]
        self.voucher_products = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in self.strikes]
        # Max qty per order
        self.limit = 200

    def implied_vol(self, S: float, K: float, T: float, market_price: float) -> float:
        """Bisection to invert BS_CALL for implied volatility."""
        low, high = 1e-6, 3.0
        for _ in range(50):
            mid = 0.5*(low + high)
            price = BS_CALL(S, K, T, 0.0, mid)
            if abs(price - market_price) < 1e-6:
                return mid
            if price > market_price:
                high = mid
            else:
                low = mid
        return mid

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. Takes all books, returns dict of orders.
        """
        result: Dict[str, List[Order]] = {}
        # 1) compute mid‐prices for everything
        mid: Dict[str, float] = {}
        for product, od in state.order_depths.items():
            if od.buy_orders and od.sell_orders:
                mb = max(od.buy_orders)
                ma = min(od.sell_orders)
                mid[product] = 0.5 * (mb + ma)

        # 2) voucher arbitrage
        day = state.timestamp // 10000
        T   = max(1e-6, (7 - day) / 7)

        S = mid.get(self.rock_product)
        if S is None:
            return result  # no underlying price

        for vp in self.voucher_products:
            od = state.order_depths.get(vp)
            mp = mid.get(vp)
            if od is None or mp is None:
                continue

            # strike from name
            K = int(vp.rsplit("_", 1)[1])
            # implied vol & fair price
            iv   = self.implied_vol(S, K, T, mp)
            fair = BS_CALL(S, K, T, 0.0, iv)
            mis  = mp - fair

            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_ask = min(od.sell_orders) if od.sell_orders else None

            # simple threshold
            if mis > 5 and best_bid:
                qty = min(self.limit, od.buy_orders[best_bid])
                if qty > 0:
                    result.setdefault(vp, []).append(Order(vp, best_bid, -qty))
            elif mis < -5 and best_ask:
                qty = min(self.limit, od.sell_orders[best_ask])
                if qty > 0:
                    result.setdefault(vp, []).append(Order(vp, best_ask, +qty))

        # no conversions or state persistence needed here
        return result, 0, ""
