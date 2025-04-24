from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState, ConversionObservation

class Trader:
    def __init__(self):
        self.macaron_limit = 75
        self.conversion_limit = 10
        self.estimated_fee = 2.0
        self.smoothing = 0.2
        self.fixed_spread = 2  # Avoid tightening dynamically
        self.mm_volume = 4     # Reasonable base size to avoid flooding book

    def run(self, state: TradingState) -> (Dict[str, List[Order]], int, str):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = "{}"

        product = "MAGNIFICENT_MACARONS"

        if product not in state.order_depths or product not in state.observations.conversionObservations:
            return result, conversions, traderData

        order_depth: OrderDepth = state.order_depths[product]
        observation: ConversionObservation = state.observations.conversionObservations[product]
        orders: List[Order] = []

        position = state.position.get(product, 0)

        # Market data
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

        # Pristine Cuisine data
        chef_bid = observation.bidPrice
        chef_ask = observation.askPrice
        transport_fee = observation.transportFees
        import_tariff = observation.importTariff
        export_tariff = observation.exportTariff

        true_buy = chef_ask + transport_fee + import_tariff
        true_sell = chef_bid - transport_fee - export_tariff

        # Update fee estimate only if no trade happened
        if best_ask and chef_bid and chef_bid - best_ask < self.estimated_fee:
            self.estimated_fee = (1 - self.smoothing) * self.estimated_fee + self.smoothing * (best_ask - chef_bid)
        if best_bid and chef_ask and best_bid - chef_ask < self.estimated_fee:
            self.estimated_fee = (1 - self.smoothing) * self.estimated_fee + self.smoothing * (chef_ask - best_bid)

        # ---------------------------------------
        # Conversion Arbitrage Logic (Priority 1)
        # ---------------------------------------

        # Buy from chef, sell to market (strong signal)
        if best_bid is not None and true_buy + self.estimated_fee + 1 < best_bid and conversions < self.conversion_limit:
            volume = min(self.macaron_limit - position, order_depth.buy_orders[best_bid])
            volume = min(volume, self.conversion_limit - conversions)
            if volume > 0:
                print("ARBITRAGE: Buy from chef & sell", volume, "x", best_bid)
                orders.append(Order(product, best_bid, -volume))
                conversions += volume

        # Buy from market, sell to chef
        if best_ask is not None and true_sell - self.estimated_fee - 1 > best_ask and conversions < self.conversion_limit:
            volume = min(self.macaron_limit + position, -order_depth.sell_orders[best_ask])
            volume = min(volume, self.conversion_limit - conversions)
            if volume > 0:
                print("ARBITRAGE: Buy", volume, "x", best_ask, "& sell to chef")
                orders.append(Order(product, best_ask, volume))
                conversions += volume

        # ----------------------------------------
        # Market Making Logic (Priority 2 - Passive)
        # ----------------------------------------

        if best_bid is not None and best_ask is not None and abs(position) < self.macaron_limit * 0.6:
            mid_price = (best_bid + best_ask) / 2
            bid_price = int(mid_price - self.fixed_spread)
            ask_price = int(mid_price + self.fixed_spread)

            if position < self.macaron_limit:
                print("MM BUY", self.mm_volume, "x", bid_price)
                orders.append(Order(product, bid_price, self.mm_volume))

            if position > -self.macaron_limit:
                print("MM SELL", self.mm_volume, "x", ask_price)
                orders.append(Order(product, ask_price, -self.mm_volume))

        result[product] = orders
        return result, conversions, traderData
