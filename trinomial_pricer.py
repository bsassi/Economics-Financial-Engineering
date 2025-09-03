"""
Trinomial option pricer implemented in Python.

This module provides a class ``TrinomialTreePricer`` that implements the
Kamrad–Ritchken recombining trinomial tree for pricing European or
American options.  It computes not only the option price but also
associated greeks (Delta, Gamma, Theta) and the transition
probabilities for up, mid, and down moves.  The pricer supports
continuous dividend yield ``q`` and can handle both calls and puts.

Example usage::

    from trinomial_pricer import TrinomialTreePricer

    # Define market and contract parameters
    S0 = 100.0     # spot price
    K  = 120.0     # strike price
    r  = 0.05      # annual risk‑free rate (decimal)
    q  = 0.0       # continuous dividend yield (decimal)
    sigma = 0.20   # annual volatility (decimal)
    T  = 0.12      # time to maturity in years
    N  = 5         # number of time steps in the tree
    is_call = True # True for call option, False for put
    is_american = False  # True for American style, False for European

    # Build the pricer and compute results
    pricer = TrinomialTreePricer(S0, K, r, q, sigma, T, N,
                                 is_call=is_call, is_american=is_american)
    result = pricer.price_option()
    print('Option price:', result.price)
    print('Delta:', result.delta)
    print('Gamma:', result.gamma)
    print('Theta:', result.theta)
    print('Probabilities: up =', result.prob_up,
          ', mid =', result.prob_mid,
          ', down =', result.prob_down)

The resulting ``TrinomialTreeResult`` object also contains the full stock and
option value lattices which can be inspected or exported for further
analysis.

"""

from dataclasses import dataclass
import math
from typing import List, Tuple

@dataclass
class TrinomialTreeResult:
    """Container for option pricing results from the trinomial tree.

    Attributes
    ----------
    price : float
        The computed option price.
    delta : float
        The first derivative of the option value with respect to the
        underlying spot price (Δ).
    gamma : float
        The second derivative of the option value with respect to the
        underlying spot price (Γ).
    theta : float
        The rate of change of the option price with respect to time (Θ).
    prob_up : float
        Probability of an up move at any node.
    prob_mid : float
        Probability of a middle (no change) move at any node.
    prob_down : float
        Probability of a down move at any node.
    stock_lattice : List[List[float]]
        A list of lists representing the stock price at each node of
        the tree.  ``stock_lattice[i][j]`` corresponds to the stock
        price at time step ``i`` and position ``j``.
    value_lattice : List[List[float]]
        A list of lists representing the option value at each node of
        the tree.  ``value_lattice[i][j]`` corresponds to the option
        value at time step ``i`` and position ``j``.
    """
    price: float
    delta: float
    gamma: float
    theta: float
    prob_up: float
    prob_mid: float
    prob_down: float
    stock_lattice: List[List[float]]
    value_lattice: List[List[float]]


class TrinomialTreePricer:
    """Kamrad–Ritchken trinomial tree option pricer.

    Parameters
    ----------
    S0 : float
        Current spot price of the underlying asset.
    K : float
        Option strike price.
    r : float
        Annual risk‑free interest rate (continuously compounded).
    q : float
        Continuous dividend yield of the underlying asset.
    sigma : float
        Annualized volatility of the underlying asset.
    T : float
        Time to maturity in years.
    N : int
        Number of time steps in the tree.
    is_call : bool, default True
        If True, price a call option; otherwise price a put.
    is_american : bool, default False
        If True, price an American style option; otherwise price
        European style.
    """

    def __init__(self,
                 S0: float,
                 K: float,
                 r: float,
                 q: float,
                 sigma: float,
                 T: float,
                 N: int,
                 *,
                 is_call: bool = True,
                 is_american: bool = False) -> None:
        if N < 1:
            raise ValueError("Number of steps N must be at least 1 for a meaningful tree.")
        self.S0 = S0
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma
        self.T = T
        self.N = N
        self.is_call = is_call
        self.is_american = is_american

        # per‑step time increment
        self.dt = T / N
        # discount factor per time step
        self.discount = math.exp(-r * self.dt)
        # movement factors (up and down)
        self.u = math.exp(sigma * math.sqrt(2.0 * self.dt))
        self.d = 1.0 / self.u
        # risk‑neutral probabilities (Kamrad–Ritchken)
        # see the derivative guidelines for details of this formula
        erdt2 = math.exp((r - q) * self.dt / 2.0)
        evol = math.exp(sigma * math.sqrt(self.dt / 2.0))
        denom = (evol - 1.0 / evol) ** 2
        self.p_up = ((erdt2 - 1.0 / evol) ** 2) / denom
        self.p_down = ((evol - erdt2) ** 2) / denom
        self.p_mid = 1.0 - self.p_up - self.p_down

    def build_stock_lattice(self) -> List[List[float]]:
        """Construct the stock price lattice.

        Returns
        -------
        List[List[float]]
            A list of time levels; each level is a list of stock prices.
        """
        levels: List[List[float]] = []
        # time 0 has a single node with price S0
        levels.append([self.S0])
        prev_center = self.S0
        # for each subsequent time step, compute the central node and
        # the up/down positions around it
        for i in range(1, self.N + 1):
            center = prev_center * math.exp((self.r - self.q) * self.dt)
            prev_center = center
            row = [center * (self.u ** k) for k in range(-i, i + 1)]
            levels.append(row)
        return levels

    def price_option(self) -> TrinomialTreeResult:
        """Price the option and compute greeks.

        Returns
        -------
        TrinomialTreeResult
            A dataclass containing the price, greeks, probabilities and
            full lattices of stock prices and option values.
        """
        stock_lattice = self.build_stock_lattice()
        # initialize the value lattice with empty lists
        value_lattice: List[List[float]] = [ [] for _ in range(self.N + 1) ]
        # payoff at maturity
        last_prices = stock_lattice[self.N]
        last_values = [max((S - self.K) if self.is_call else (self.K - S), 0.0)
                       for S in last_prices]
        value_lattice[self.N] = last_values
        # backward recursion
        for step in range(self.N - 1, -1, -1):
            level_values = []
            for idx in range(len(stock_lattice[step])):
                # children values from next level: down, mid and up
                v_down = value_lattice[step + 1][idx]
                v_mid  = value_lattice[step + 1][idx + 1]
                v_up   = value_lattice[step + 1][idx + 2]
                hold_value = self.discount * (self.p_up * v_up +
                                              self.p_mid * v_mid +
                                              self.p_down * v_down)
                # intrinsic value if exercised now
                stock_price = stock_lattice[step][idx]
                intrinsic = (stock_price - self.K) if self.is_call else (self.K - stock_price)
                intrinsic = max(intrinsic, 0.0)
                if self.is_american:
                    level_values.append(max(hold_value, intrinsic))
                else:
                    level_values.append(hold_value)
            value_lattice[step] = level_values
        # compute greeks using the first time step
        # values at level 1 correspond to three nodes: down, mid, up
        S_down, S_mid, S_up = stock_lattice[1][:3]
        V_down, V_mid, V_up = value_lattice[1][:3]
        delta = (V_up - V_down) / (S_up - S_down)
        gamma = (((V_up - V_mid) / (S_up - S_mid)) -
                 ((V_mid - V_down) / (S_mid - S_down))) / ((S_up - S_down) / 2.0)
        theta = (V_mid - value_lattice[0][0]) / self.dt
        return TrinomialTreeResult(
            price=value_lattice[0][0],
            delta=delta,
            gamma=gamma,
            theta=theta,
            prob_up=self.p_up,
            prob_mid=self.p_mid,
            prob_down=self.p_down,
            stock_lattice=stock_lattice,
            value_lattice=value_lattice
        )


if __name__ == "__main__":
    # Example execution when run as a script
    # default parameter values mirror those used in the macro example
    S0 = 89.0
    K  = 110.0
    r  = 0.03
    q  = 0.0
    sigma = 0.24
    T  = 0.24
    N  = 9
    is_call = True
    is_american = False
    pricer = TrinomialTreePricer(S0, K, r, q, sigma, T, N,
                                 is_call=is_call, is_american=is_american)
    result = pricer.price_option()
    print(f"S0 = {S0}, K = {K}, r = {r}, q = {q}, sigma = {sigma}, T = {T}, N = {N}")
    print(f"Call option price: {result.price:.10f}")
    print(f"Delta: {result.delta:.10f}")
    print(f"Gamma: {result.gamma:.10f}")
    print(f"Theta: {result.theta:.10f}")
    print(f"Probabilities -> up: {result.prob_up:.6f}, mid: {result.prob_mid:.6f}, down: {result.prob_down:.6f}")