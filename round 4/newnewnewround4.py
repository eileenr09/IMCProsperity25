from typing import Dict, List
from datamodel import Order, OrderDepth, TradingState, ConversionObservation

class Trader:
    def __init__(self):
        self.macaron_limit = 75
        self.conversion_limit = 10
        self.estimated_fee = 2.0
        self.smoothing = 0.2
        self.spread = 2
        self.last_position = 0  # for stuck detection

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

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

        chef_bid = observation.bidPrice
        chef_ask = observation.askPrice
        transport_fee = observation.transportFees
        import_tariff = observation.importTariff
        export_tariff = observation.exportTariff

        true_buy = chef_ask + transport_fee + import_tariff
        true_sell = chef_bid - transport_fee - export_tariff

        # Update estimated fee only if no arb possible
        if best_ask and chef_bid and chef_bid - best_ask < self.estimated_fee:
            self.estimated_fee = (1 - self.smoothing) * self.estimated_fee + self.smoothing * (best_ask - chef_bid)
        if best_bid and chef_ask and best_bid - chef_ask < self.estimated_fee:
            self.estimated_fee = (1 - self.smoothing) * self.estimated_fee + self.smoothing * (chef_ask - best_bid)

        # Adaptive conversion batch size (lower near limit)
        safe_conversion = min(6, max(1, self.macaron_limit - abs(position)))

        # -------------------------------
        # Conversion Arbitrage Logic
        # -------------------------------
        if best_bid is not None and true_buy + self.estimated_fee + 0.5 < best_bid and conversions < self.conversion_limit:
            volume = min(safe_conversion, order_depth.buy_orders[best_bid])
            volume = min(volume, self.conversion_limit - conversions)
            if volume > 0:
                orders.append(Order(product, best_bid, -volume))
                conversions += volume

        if best_ask is not None and true_sell - self.estimated_fee - 0.5 > best_ask and conversions < self.conversion_limit:
            volume = min(safe_conversion, -order_depth.sell_orders[best_ask])
            volume = min(volume, self.conversion_limit - conversions)
            if volume > 0:
                orders.append(Order(product, best_ask, volume))
                conversions += volume

        # -------------------------------
        # Market Making Logic
        # -------------------------------
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2

            # Wider spread if near limits
            buffer_spread = self.spread + int(abs(position) > self.macaron_limit * 0.7)
            bid_price = int(mid_price - buffer_spread)
            ask_price = int(mid_price + buffer_spread)

            buy_volume = min(5, self.macaron_limit - position)
            sell_volume = min(5, self.macaron_limit + position)

            if buy_volume > 0:
                orders.append(Order(product, bid_price, buy_volume))
            if sell_volume > 0:
                orders.append(Order(product, ask_price, -sell_volume))

        # Failsafe: if stuck in same position for too long, aggressively flatten
        if abs(position) >= self.macaron_limit - 5 and position == self.last_position:
            flatten_price = best_ask if position > 0 else best_bid
            flatten_volume = min(5, abs(position))
            if position > 0:
                orders.append(Order(product, best_ask, -flatten_volume))  # sell to flatten
            elif position < 0:
                orders.append(Order(product, best_bid, flatten_volume))   # buy to flatten

        self.last_position = position

        result[product] = orders
        return result, conversions, traderData
