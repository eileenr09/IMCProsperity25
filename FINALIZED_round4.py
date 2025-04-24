from typing import Dict, List
from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List
import string
import jsonpickle
import numpy as np
import math
from statistics import NormalDist
import statistics

# Option
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
MACARON = "MAGNIFICENT_MACARONS"

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
        self.recipes = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2},
        }

        self.limits = {RESIN: 50,
            KELP: 50,
            SQUID: 50,
            "CROISSANTS": 250,
            "JAMS":   350,
            "DJEMBES":    60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2":100
        }
        self.price_history = {
            SQUID: [],
            PICNIC_BASKET1: [],
            PICNIC_BASKET2: []
        }

        # SQUID PARAMS
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

        self.tick_count = 0

        # Volcanos
        # we only trade these three voucher strikes
        self.strikes = [10_000, 10_250, 10_500]
        self.voucher_products = [f"VOLCANIC_ROCK_VOUCHER_{k}" for k in self.strikes]
        self.rock_product     = "VOLCANIC_ROCK"
        self.limit            = 200
        # mispricing history (to adapt threshold)
        self.mis_history: Dict[int, List[float]] = {k: [] for k in self.strikes}
        self.hist_len = 50

        # Macaron
        self.macaron_limit = 75
        self.conversion_limit = 10
        self.estimated_fee = 2.0
        self.smoothing = 0.2
        self.spread = 2  # spread around midprice for market making


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
        position_limit = self.limits[product]

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
        buy_quantity = self.limits[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.limits[product] + (position - sell_order_volume)
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

        buy_quantity = self.limits[product] - (position + buy_order_volume)
        sell_quantity = self.limits[product] + (position - sell_order_volume)

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
        limit = self.limits[product]

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
        limit = self.limits[product]
        position = positions.get(product, 0)

        buy_volume = min(MIN_VOLUME, limit - position)
        sell_volume = min(MIN_VOLUME, position + limit)

        if buy_volume > 0:
            orders.append(Order(product, buy_price, buy_volume))

        if sell_volume > 0:
            orders.append(Order(product, sell_price, -sell_volume))
        return orders

    # Options
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



    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        traderObject = {}

        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        result = {}
        conversions = 0

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
        od: OrderDepth = state.order_depths.get(SQUID)
        orders: List[Order] = []

        if SQUID in order_depths:
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


        # Volcanos

         # 1) compute mid‐prices
        mid: Dict[str,float] = {}
        for sym, od in state.order_depths.items():
            if od.buy_orders and od.sell_orders:
                mb, ma = max(od.buy_orders), min(od.sell_orders)
                mid[sym] = 0.5*(mb + ma)

        # 2) underlying spot
        S = mid.get(self.rock_product)
        if S:
            # return result, conversions, state.traderData
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

        # Macaron
        if MACARON in state.order_depths and MACARON in state.observations.conversionObservations:
        
            product = MACARON
            order_depth: OrderDepth = state.order_depths[product]
            observation: ConversionObservation = state.observations.conversionObservations[product]

            orders: List[Order] = []

            position = state.position.get(product, 0)

            # Market data
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

            # Chef conversion prices
            chef_bid = observation.bidPrice
            chef_ask = observation.askPrice
            transport_fee = observation.transportFees
            import_tariff = observation.importTariff
            export_tariff = observation.exportTariff

            # True prices for buy/sell
            true_buy = chef_ask + transport_fee + import_tariff
            true_sell = chef_bid - transport_fee - export_tariff

            # Dynamically estimate fees if no arbitrage seen
            if best_ask is not None and chef_bid - best_ask < self.estimated_fee:
                self.estimated_fee = (1 - self.smoothing) * self.estimated_fee + self.smoothing * (best_ask - chef_bid)

            if best_bid is not None and best_bid - chef_ask < self.estimated_fee:
                self.estimated_fee = (1 - self.smoothing) * self.estimated_fee + self.smoothing * (chef_ask - best_bid)

            # -------------------------------
            # Arbitrage Logic via Conversion
            # -------------------------------

            # Buy from chef, sell to market
            if best_bid is not None and true_buy + self.estimated_fee < best_bid and conversions < self.conversion_limit:
                volume = min(self.macaron_limit - position, order_depth.buy_orders[best_bid])
                volume = min(volume, self.conversion_limit - conversions)
                if volume > 0:
                    print("ARBITRAGE: Buy from chef & sell", volume, "x", best_bid)
                    orders.append(Order(product, best_bid, -volume))
                    conversions += volume

            # Buy from market, sell to chef
            if best_ask is not None and true_sell - self.estimated_fee > best_ask and conversions < self.conversion_limit:
                volume = min(self.macaron_limit + position, -order_depth.sell_orders[best_ask])
                volume = min(volume, self.conversion_limit - conversions)
                if volume > 0:
                    print("ARBITRAGE: Buy", volume, "x", best_ask, "& sell to chef")
                    orders.append(Order(product, best_ask, volume))
                    conversions += volume

            # ---------------------
            # Market Making Logic
            # ---------------------
            if best_bid is not None and best_ask is not None:
                mid_price = (best_bid + best_ask) / 2
                buy_price = int(mid_price - self.spread)
                sell_price = int(mid_price + self.spread)

                buy_volume = min(5, self.macaron_limit - position)
                sell_volume = min(5, self.macaron_limit + position)

                if buy_volume > 0:
                    print("MM BUY", buy_volume, "x", buy_price)
                    orders.append(Order(product, buy_price, buy_volume))

                if sell_volume > 0:
                    print("MM SELL", sell_volume, "x", sell_price)
                    orders.append(Order(product, sell_price, -sell_volume))

            # Final order list
            result[product] = orders
      
        traderData = jsonpickle.encode(traderObject)
        
        return result, conversions, traderData