from typing import Dict, List, Tuple
import math
import jsonpickle
from datamodel import OrderDepth, TradingState, Order

SQUID = "SQUID_INK"
class Trader:
    def __init__(self):
        # EWMA parameters for mid‑price and variance
        self.alpha_squid = 0.2
        # Market‑making parameters
        self.spread_multiplier_squid = 2.0   # half‑spread = multiplier × σ
        self.min_spread_squid        = 1     # minimum half‑spread
        self.max_spread_squid      = 10    # maximum half‑spread
        self.base_order_size_squid   = 20    # nominal size per quote
        self.min_order_size_squid    = 5
        self.max_order_size_squid    = 50
        self.inventory_risk_squid    = 0.1   # ticks skew per unit

        # Persistent EWMA state
        self.ewma_mid_squid = None
        self.ewma_var_squid = None

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # load EWMA state
        if state.traderData:
            saved = jsonpickle.loads(state.traderData)
            self.ewma_mid_squid = saved.get("ewma_mid", self.ewma_mid_squid)
            self.ewma_var_squid = saved.get("ewma_var", self.ewma_var_squid)

        od: OrderDepth = state.order_depths.get(SQUID)
        orders: List[Order] = []

        if od and od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid = 0.5 * (best_bid + best_ask)

            # initialize EWMA on first tick
            if self.ewma_mid_squid is None:
                self.ewma_mid_squid = mid
                self.ewma_var_squid = 1.0  # originally at 1

            # update EWMA mid and variance
            diff = mid - self.ewma_mid_squid
            self.ewma_mid_squid += self.alpha_squid * diff
            self.ewma_var_squid = (1 - self.alpha_squid) * self.ewma_var_squid + self.alpha_squid * (diff * diff)

            sigma = math.sqrt(self.ewma_var_squid)

            # compute dynamic half‑spread
            half_spread = self.spread_multiplier_squid * sigma
            half_spread = max(self.min_spread_squid, min(self.max_spread_squid, half_spread))

            # inventory skew
            pos = state.position.get(SQUID, 0)
            skew = self.inventory_risk_squid * pos

            # dynamic order size: shrink when inventory is large
            inv_ratio = abs(pos) / self.max_order_size_squid
            size = int(self.base_order_size_squid * (1 - inv_ratio))
            size = max(self.min_order_size_squid, min(self.max_order_size_squid, size))

            # quote prices
            bid_px = int(round(self.ewma_mid_squid - half_spread - skew))
            ask_px = int(round(self.ewma_mid_squid + half_spread - skew))
            if bid_px >= ask_px:
                bid_px = max(1, ask_px - 1)

            # emit quotes
            orders = [
                Order(SQUID, bid_px,  size),
                Order(SQUID, ask_px, -size)
            ]

        result[SQUID] = orders

        # persist state
        traderData = jsonpickle.dumps({
            "ewma_mid": self.ewma_mid_squid,
            "ewma_var": self.ewma_var_squid
        })
        return result, conversions, traderData
