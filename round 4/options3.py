from typing import Dict, List, Tuple
import math
import statistics
import jsonpickle
from datamodel import Order, OrderDepth, TradingState
from statistics import NormalDist

def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x*x) / math.sqrt(2*math.pi)

def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black–Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    N = NormalDist().cdf
    return S * N(d1) - K * math.exp(-r*T) * N(d2)

def greek_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Δ = ∂C/∂S."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    return NormalDist().cdf(d1)

def greek_gamma(S: float, K: float, T: float, sigma: float) -> float:
    """Γ = ∂²C/∂S²."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S/K) + 0.5*sigma*sigma*T) / (sigma*math.sqrt(T))
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))

def greek_vega(S: float, K: float, T: float, sigma: float) -> float:
    """Vega = ∂C/∂σ."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S/K) + 0.5*sigma*sigma*T) / (sigma*math.sqrt(T))
    return S * norm_pdf(d1) * math.sqrt(T)

def greek_theta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Θ = ∂C/∂t (per year)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    pdf_d1 = norm_pdf(d1)
    N = NormalDist().cdf
    term1 = - (S * pdf_d1 * sigma) / (2 * math.sqrt(T))
    term2 = - r * K * math.exp(-r*T) * N(d2)
    return term1 + term2

class Trader:
    def __init__(self):
        # we only trade these three voucher strikes
        self.strikes = [10_000, 10_250, 10_500]
        self.voucher_products = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in self.strikes]
        self.rock_product     = "VOLCANIC_ROCK"
        self.limit            = 200
        # mispricing history (to adapt threshold)
        self.mis_history: Dict[int, List[float]] = {k: [] for k in self.strikes}
        self.hist_len = 50

    def implied_vol(self, S: float, K: float, T: float, mp: float) -> float:
        """Invert BS for IV by bisection."""
        low, high = 1e-6, 3.0
        for _ in range(50):
            mid = 0.5*(low + high)
            price = bs_call_price(S, K, T, mid)
            if abs(price - mp) < 1e-6:
                return mid
            if price > mp:
                high = mid
            else:
                low = mid
        return mid

    def run(self, state: TradingState) -> Tuple[Dict[str,List[Order]], int, str]:
        result: Dict[str,List[Order]] = {}
        conversions = 0
        traderData = ""  # no persistence needed here

        # 1) compute mid‐prices
        mid: Dict[str,float] = {}
        for sym, od in state.order_depths.items():
            if od.buy_orders and od.sell_orders:
                mb, ma = max(od.buy_orders), min(od.sell_orders)
                mid[sym] = 0.5*(mb + ma)

        # 2) underlying spot
        S = mid.get(self.rock_product)
        if S is None:
            return result, conversions, traderData

        day = state.timestamp // 10000
        T   = max(1e-6, (7-day)/7)

        net_delta = 0.0
        orders: List[Order] = []

        # 3) arbitrage + greek‐sized voucher trades
        for K in self.strikes:
            vp = f"VOLCANIC_ROCK_VOUCHER_{K}"
            od = state.order_depths.get(vp)
            mp = mid.get(vp)
            if od is None or mp is None:
                continue

            iv    = self.implied_vol(S, K, T, mp)
            fair  = bs_call_price(S, K, T, iv)
            mis   = mp - fair

            # update mispricing history
            hist = self.mis_history[K]
            hist.append(mis)
            if len(hist) > self.hist_len:
                hist.pop(0)
            # adaptive threshold = max(3, 1.5×stdev)
            thr = 5.0
            if len(hist) >= 5:
                thr = max(3.0, 1.5 * statistics.stdev(hist))

            # compute Greeks
            Δ = greek_delta(S, K, T, iv)
            Γ = greek_gamma(S, K, T, iv)
            Θ = greek_theta(S, K, T, iv)
            ν = greek_vega(S, K, T, iv)

            # choose side and size
            bb = max(od.buy_orders)  if od.buy_orders  else None
            ba = min(od.sell_orders) if od.sell_orders else None

            # LONG voucher if mispricing very negative
            if mis < -thr:
                size = min(self.limit, od.sell_orders[ba])
                # scale by vega (more vega→smaller size to control vol risk)
                size = max(1, int(size / (1+ν*5)))
                if size>0:
                    orders.append(Order(vp, ba, +size))
                    net_delta += Δ * size

            # SHORT voucher if mispricing very positive
            elif mis > +thr:
                size = min(self.limit, od.buy_orders[bb])
                size = max(1, int(size / (1+ν*5)))
                if size>0:
                    orders.append(Order(vp, bb, -size))
                    net_delta -= Δ * size

        # 4) delta‐hedge the rock
        if abs(net_delta) > 0.5 and self.rock_product in state.order_depths:
            odr = state.order_depths[self.rock_product]
            rb  = max(odr.buy_orders)  if odr.buy_orders  else None
            ra  = min(odr.sell_orders) if odr.sell_orders else None
            pos = state.position.get(self.rock_product, 0)
            target = -net_delta
            diff   = target - pos

            if diff > 0 and ra:
                vol = min(diff, odr.sell_orders[ra], 400 - pos)
                if vol>0:
                    orders.append(Order(self.rock_product, ra, +round(vol)))
            elif diff < 0 and rb:
                vol = min(-diff, odr.buy_orders[rb], 400 + pos)
                if vol>0:
                    orders.append(Order(self.rock_product, rb, -round(vol)))

        # bucket by symbol
        for o in orders:
            result.setdefault(o.symbol, []).append(o)

        return result, conversions, traderData
