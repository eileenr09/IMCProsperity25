from typing import Dict, List
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import math
import statistics

outer_limit = 50
LIMIT_BSK1 = 60
LIMIT_BSK2 = 100

PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"

RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID = "SQUID_INK"
CROISSANTS = "CROISSANTS"
JAMS = "JAMS"
DJEMBE = "DJEMBE"

PARAMS = {
    KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
}

THRESHOLD_HIGH = 0.8
THRESHOLD_LOW = -0.8
PRICE_OFFSET = 1  # Shift bid/ask slightly to increase chance of execution
ALPHA = 0.1
VOL_WINDOW = 20

LIMITS = {RESIN: 50, KELP: 50, SQUID: 50, PICNIC_BASKET1: LIMIT_BSK1, PICNIC_BASKET2: LIMIT_BSK2}


K = {
    RESIN: 1,
    SQUID: 1,
    PICNIC_BASKET1: 1,
    PICNIC_BASKET2: 1,
    KELP: 0.5,
}

MIN_VOLUME = 5  # Minimum volume to trade
BASE_SPREAD = 1.0  # Base price offset to increase fill chance
SPREAD_EDGE = 1  # Offset to increase fill chance

class Trader:
    def __init__(self, params=None):
        self.ema = {}
        self.ma = {}
        self.volatility = {}
        self.price_history_volatility = {}
        self.price_history_ma = {}
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {RESIN: 50,
            KELP: 50,
            SQUID: 50}
        self.price_history = {
            SQUID: [],
            PICNIC_BASKET1: [],
            PICNIC_BASKET2: []
        }
        self.window_size_sq = 5
        self.std_dev_multiplier = 0.85
        self.stop_loss_multiplier = 2.75
        self.max_trade_duration = 35
        
        self.tick_count = 0

        # Volcanos
        self.voucher_pnl = 0.0
        self.strikes = [9500, 9750, 10000, 10250, 10500]
        self.voucher_products = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in self.strikes]
        self.basic_products = ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]
        self.rock_product = "VOLCANIC_ROCK"
        self.window_size_vol = 5
        self.limit_vol = 200
        self.basic_limits = {p: 50 for p in self.basic_products}
        # Boost SQUID_INK exposure dynamically on Day 0
        self.default_squid_limit = 100
        self.day0_squid_limit = 200  # Stronger allocation on Day 0
        self.tick_count_vol = 0
        self.price_history_vol = {p: [] for p in self.basic_products}
        self.active_trades = {p: None for p in self.basic_products}
        self.active_trades_squid = {SQUID: None}
        # self.active_trades = {p: None for p in self.basic_products}
        self.rock_prices = []  # for adaptive volatility estimation

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
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
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

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
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

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def take_orders_kelp(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
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
    ) -> (List[Order], int, int):
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

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_ask + best_bid) / 2

    def get_best_bid_ask(self, order_depth):
        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)
        return best_bid, best_ask
    
    def update_ema(self, product: str, mid: float):
        if product not in self.ema:
            self.ema[product] = mid
        else:
            self.ema[product] = ALPHA * mid + (1 - ALPHA) * self.ema[product]
    
    def update_price_history(self, product: str, mid: float):
        if product not in self.price_history:
            self.price_history[product] = []
        self.price_history[product].append(mid)
        if len(self.price_history[product]) > VOL_WINDOW:
            self.price_history[product].pop(0)
    
    def update_ma(self, product: str, mid: float):
        window = SQUID_MA_WINDOW
        if product not in self.price_history_ma:
            self.price_history_ma[product] = []
        self.price_history_ma[product].append(mid)
        if len(self.price_history_ma[product]) > window:
            self.price_history_ma[product].pop(0)
        self.ma[product] = sum(self.price_history_ma[product]) / len(self.price_history_ma[product])

    def get_volatility(self, product: str):
        if product not in self.price_history or len(self.price_history[product]) < 2:
            return 1
        hist = self.price_history[product]
        mean = sum(hist) / len(hist)
        return (sum((x - mean) ** 2 for x in hist) / len(hist)) ** 0.5

    def update_volatility(self, product: str, mid: float):
        window = VOL_WINDOW
        if product not in self.price_history_volatility:
            self.price_history_volatility[product] = []
        self.price_history_volatility[product].append(mid)
        if len(self.price_history_volatility[product]) > window:
            self.price_history_volatility[product].pop(0)
        if len(self.price_history_volatility[product]) >= 2:
            diffs = [(p - self.ema[product]) ** 2 for p in self.price_history_volatility[product]]
            self.volatility[product] = (sum(diffs) / len(diffs)) ** 0.5
        else:
            self.volatility[product] = 1.0  # Avoid divide-by-zero
    
    def compute_etf_fair(self, state: TradingState, basket: str):
        mids = {}
        for prod in [CROISSANTS, JAMS, DJEMBE]:
            if prod in state.order_depths:
                mids[prod] = self.get_mid_price(state.order_depths[prod])
        if CROISSANTS not in mids or JAMS not in mids:
            return None
        if basket == PICNIC_BASKET1:
            return 6 * mids[CROISSANTS] + 3 * mids[JAMS] + (mids.get(DJEMBE, 0) or 0)
        elif basket == PICNIC_BASKET2:
            return 4 * mids[CROISSANTS] + 2 * mids[JAMS]
        return None


    def take_orders_resin(self, state, product, fair_price, vol):
        k = K[product]
        positions = state.position
        order_depth = state.order_depths[product]
        position = positions.get(product, 0)
        limit = LIMITS[product]

        orders = []

        # Buy logic
        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if ask_price < fair_price - k*vol:
                buy_volume = min(abs(ask_volume), limit - position, MIN_VOLUME)
                if buy_volume > 0:
                    orders.append(Order(product, ask_price, buy_volume))
                    position += buy_volume
            else:
                break

        # Sell logic
        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid_price > fair_price + k*vol:
                sell_volume = min(abs(bid_volume), limit + position, MIN_VOLUME)
                if sell_volume > 0:
                    orders.append(Order(product, bid_price, -sell_volume))
                    position -= sell_volume
            else:
                break
        return orders
    
    def make_market(self, state, product, fair_price, edge):
        positions = state.position
        orders = []
        buy_price = int(fair_price - edge)
        sell_price = int(fair_price + edge)
        limit = LIMITS[product]
        position = positions.get(product, 0)

        buy_volume = min(MIN_VOLUME, limit - position)
        sell_volume = min(MIN_VOLUME, position + limit)

        if buy_volume > 0:
            orders.append(Order(product, buy_price, buy_volume))

        if sell_volume > 0:
            orders.append(Order(product, sell_price, -sell_volume))
        return orders

    def get_fair_value_and_bands(self, product, order_depth):
        current_price = self.get_mid_price(order_depth)
        if current_price is None:
            return None, None, None, None, None

        history = self.price_history[product]
        history.append(current_price)
        if len(history) > self.window_size_sq:
            history.pop(0)

        sma = sum(history) / len(history)
        std_dev = statistics.stdev(history) if len(history) > 1 else 0

        upper_band = sma + self.std_dev_multiplier * std_dev
        lower_band = sma - self.std_dev_multiplier * std_dev

        return sma, upper_band, lower_band, current_price, std_dev
    def execute_trade_squid(self, product, fair_value, upper_band, lower_band, current_price, std_dev, order_depth, state):
        orders = []

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return orders

        position = state.position.get(product, 0)
        limit = self.LIMIT[product]
        trade = self.active_trades_squid[product]
        # volatility_scale = max(0.1, 1 / (1 + std_dev))  # Prevents scaling too small
        # adjusted_limit = int(limit * volatility_scale)
        adjusted_limit = limit

        # OPEN trade
        if trade is None:
            if current_price > upper_band:
                sell_qty = min(position + adjusted_limit, outer_limit)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    self.active_trades_squid[product] = {
                        "position": "short",
                        "target_price": fair_value,
                        "entry_tick": self.tick_count,
                        "stop_loss": current_price + self.stop_loss_multiplier * std_dev
                    }

            elif current_price < lower_band:
                buy_qty = min(adjusted_limit - position, outer_limit)
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    self.active_trades_squid[product] = {
                        "position": "long",
                        "target_price": fair_value,
                        "entry_tick": self.tick_count,
                        "stop_loss": current_price - self.stop_loss_multiplier * std_dev
                    }

        # CLOSE trade
        elif trade:
            time_in_trade = self.tick_count - trade["entry_tick"]
            stop_loss = trade["stop_loss"]
            exit_due_to_timeout = time_in_trade >= self.max_trade_duration

            if trade["position"] == "long":
                if current_price >= trade["target_price"] or current_price <= stop_loss or exit_due_to_timeout:
                    sell_qty = min(position, limit)
                    if sell_qty > 0:
                        orders.append(Order(product, best_bid, -sell_qty))
                        self.active_trades_squid[product] = None

            elif trade["position"] == "short":
                if current_price <= trade["target_price"] or current_price >= stop_loss or exit_due_to_timeout:
                    buy_qty = min(limit + position, limit)
                    if buy_qty > 0:
                        orders.append(Order(product, best_ask, buy_qty))
                        self.active_trades_squid[product] = None

        return orders

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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}

        order_depths = state.order_depths

        # RESIN
        if RESIN in order_depths:
            edge = 3
            fair_value = 10000
            take_orders_resin = self.take_orders_resin(state, RESIN, fair_value, 0)
            make_orders = self.make_market(state, RESIN, fair_value, edge)
            # make_orders = []
            orders = take_orders_resin + make_orders
            result[RESIN] = orders

        # SQUID
        if SQUID in order_depths:
          self.tick_count += 1
          order_depth = state.order_depths.get(SQUID)
          if order_depth:
            fair_value, upper_band, lower_band, current_price, std_dev = self.get_fair_value_and_bands(SQUID, order_depth)
            if fair_value:
              orders = self.execute_trade_squid(SQUID, fair_value, upper_band, lower_band, current_price, std_dev, order_depth, state)
              result[SQUID] = orders

        # KELP
        if KELP in self.params and KELP in state.order_depths:
            KELP_position = (
                state.position[KELP]
                if KELP in state.position
                else 0
            )
            KELP_fair_value = self.kelp_fair_value(
                state.order_depths[KELP], traderObject
            )
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders_kelp(
                    KELP,
                    state.order_depths[KELP],
                    KELP_fair_value,
                    self.params[KELP]["take_width"],
                    KELP_position,
                    self.params[KELP]["prevent_adverse"],
                    self.params[KELP]["adverse_volume"],
                )
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    KELP,
                    state.order_depths[KELP],
                    KELP_fair_value,
                    self.params[KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            KELP_make_orders, _, _ = self.make_orders(
                KELP,
                state.order_depths[KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[KELP]["disregard_edge"],
                self.params[KELP]["join_edge"],
                self.params[KELP]["default_edge"],
            )
            result[KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )
        
        # ETF
        for product in [PICNIC_BASKET1, PICNIC_BASKET2]:
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            mid = self.get_mid_price(order_depth)
            if mid is None:
                continue

            self.update_ema(product, mid)
            self.update_price_history(product, mid)
            vol = self.get_volatility(product)

            fair = self.compute_etf_fair(state, product)
            if fair is None:
                continue

            zscore = (mid - fair) / vol if vol > 0 else 0
            orders = []
            limit = LIMIT_BSK1 if product == PICNIC_BASKET1 else LIMIT_BSK2
            pos = state.position.get(product, 0)

            # Buy if underpriced relative to ETF fair value
            if zscore < THRESHOLD_LOW:
                best_ask = min(order_depth.sell_orders)
                buy_price = best_ask + PRICE_OFFSET
                buy_volume = min(abs(order_depth.sell_orders[best_ask]), limit - pos)
                if buy_volume > 0:
                    orders.append(Order(product, buy_price, buy_volume))

            # Sell if overpriced
            if zscore > THRESHOLD_HIGH:
                best_bid = max(order_depth.buy_orders)
                sell_price = best_bid - PRICE_OFFSET
                sell_volume = min(abs(order_depth.buy_orders[best_bid]), limit + pos)
                if sell_volume > 0:
                    orders.append(Order(product, sell_price, -sell_volume))

            result[product] = orders


        # Volcanos
        self.tick_count_vol += 1
        # result = {}
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

            self.price_history_vol[product].append(current_price)
            if len(self.price_history_vol[product]) > self.window_size_vol:
                self.price_history_vol[product].pop(0)

            if len(self.price_history_vol[product]) < 2:
                continue

            sma = sum(self.price_history_vol[product]) / len(self.price_history_vol[product])
            std = statistics.stdev(self.price_history_vol[product])
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
                recent_prices = self.price_history_vol[product][-self.window_size_vol:]
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
                recent_prices = self.price_history_vol[product][-self.window_size_vol:]
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
                        self.active_trades[product] = {"position": "long", "entry_tick": self.tick_count_vol, "target_price": sma, "stop_loss": current_price - stop_mult * std, "max_hold": max_hold}
                elif current_price > upper_band and best_bid:
                    qty = min(position + limit, order_depth.buy_orders[best_bid])
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                        self.active_trades[product] = {"position": "short", "entry_tick": self.tick_count_vol, "target_price": sma, "stop_loss": current_price + stop_mult * std, "max_hold": max_hold}
            else:
                trade = self.active_trades[product]
                time_in_trade = self.tick_count_vol - trade["entry_tick"]
                max_hold = trade.get("max_hold", self.max_trade_duration)
                if trade["position"] == "long":
                    if current_price >= trade["target_price"] or current_price <= trade["stop_loss"] or time_in_trade >= max_hold:
                        if best_bid:
                            qty = min(position, order_depth.buy_orders[best_bid])
                            if qty > 0:
                                orders.append(Order(product, best_bid, -qty))
                                self.active_trades[product] = None
                elif trade["position"] == "short":
                    if current_price <= trade["target_price"] or current_price >= trade["stop_loss"] or time_in_trade >= max_hold:
                        if best_ask:
                            qty = min(limit + position, order_depth.sell_orders[best_ask])
                            if qty > 0:
                                orders.append(Order(product, best_ask, qty))
                                self.active_trades[product] = None

            # if orders:
            #     if result.get(product):  # If result[product] is not empty
            #         result[product].extend(orders)  # Extend it by orders
            #     else:
            #         result[product] = orders 

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
                        qty = int(self.limit_vol * min(1.0, mispricing / qty_scale_factor))
                        if qty * best_bid > max_loss_per_voucher:
                            qty = max_loss_per_voucher // best_bid
                        qty = min(qty, order_depth.buy_orders[best_bid])
                        orders.append(Order(product, best_bid, -qty))
                        self.voucher_pnl += qty * best_bid
                    elif mispricing < -mispricing_thresh and best_ask:
                        qty = int(self.limit_vol * min(1.0, abs(mispricing) / qty_scale_factor))
                        if qty * best_ask > max_loss_per_voucher:
                            qty = max_loss_per_voucher // best_ask
                        qty = min(qty, order_depth.sell_orders[best_ask])
                        orders.append(Order(product, best_ask, qty))
                        self.voucher_pnl -= qty * best_ask

                    if orders:
                        result[product] = orders


        traderData = jsonpickle.encode(traderObject)
        conversions = 1 
        
        return result, conversions, traderData





