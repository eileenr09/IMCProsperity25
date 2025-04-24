from typing import Dict, List
import math
from datamodel import OrderDepth, TradingState, Order

class Trader:
    def __init__(self):
        # “Recipes” for each basket
        self.recipes = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2},
        }

        # Position limits
        self.limits = {
            "CROISSANTS": 250,
            "JAMS":      350,
            "DJEMBES":    60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2":100,
        }

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}

        # 1) Extract best bids/asks for everything
        mid_bid = {}
        mid_ask = {}
        for sym, od in state.order_depths.items():
            if od.buy_orders:
                mid_bid[sym] = max(od.buy_orders)
            if od.sell_orders:
                mid_ask[sym] = min(od.sell_orders)

        # 2) Basket arbitrage
        for basket, recipe in self.recipes.items():
            # need both sides of the basket book
            if basket not in mid_bid or basket not in mid_ask:
                continue

            bb = mid_bid[basket]
            ba = mid_ask[basket]

            # compute synthetic cost to build one basket from components
            synth_cost = 0.0
            for ingr, qty in recipe.items():
                if ingr not in mid_ask:
                    synth_cost = None
                    break
                synth_cost += qty * mid_ask[ingr]
            if synth_cost is None:
                continue

            # compute synthetic value if we were to break basket into components
            synth_value = 0.0
            for ingr, qty in recipe.items():
                if ingr not in mid_bid:
                    synth_value = None
                    break
                synth_value += qty * mid_bid[ingr]
            if synth_value is None:
                continue

            # threshold of 1 tick to avoid ping‑pong
            tick = 1

            # (A) BUY low‐priced basket & SELL components
            if ba + tick < synth_value:
                # 1 basket
                result.setdefault(basket, []).append(Order(basket, ba, +1))
                # sell each ingredient
                for ingr, qty in recipe.items():
                    result.setdefault(ingr, []).append(
                        Order(ingr, mid_bid[ingr], -qty)
                    )

            # (B) SELL expensive basket & BUY components
            if bb - tick > synth_cost:
                # 1 basket
                result.setdefault(basket, []).append(Order(basket, bb, -1))
                # buy each ingredient
                for ingr, qty in recipe.items():
                    result.setdefault(ingr, []).append(
                        Order(ingr, mid_ask[ingr], +qty)
                    )

        # 3) (Optional) basic market‐making on the three ingredients
        #    e.g. simple top‐of‐book quotes to earn spread
        for prod in ["CROISSANTS", "JAMS", "DJEMBES"]:
            if prod in mid_bid and prod in mid_ask:
                bb = mid_bid[prod]
                ba = mid_ask[prod]
                # 1‑tick inside the NBBO
                bid_px = bb + 1
                ask_px = ba - 1
                if bid_px < ask_px:
                    size = min(10, self.limits[prod] - abs(state.position.get(prod, 0)))
                    result.setdefault(prod, []).append(Order(prod, bid_px,  +size))
                    result.setdefault(prod, []).append(Order(prod, ask_px, -size))

        return result, 0, ""
