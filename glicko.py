from collections import defaultdict
import numpy as np
import math

GLICKO_MAX = 7000
GLICKO_MIN = 0

DEFAULT_RATING = 1500
DEFAULT_RD = 350
DEFAULT_VOLATILITY = 0.06
TAU = 0.5


class GlickoRater:

    def __init__(
        self,
        initial_rating: int = DEFAULT_RATING,
        initial_rd: int = DEFAULT_RD,
        initial_volatility: float = DEFAULT_VOLATILITY,
        tau: float = TAU,
        N: int = 256
    ):
        self.ratings = {uid: initial_rating for uid in range(0, N+1)}
        self.rd = {uid: initial_rd for uid in range(0, N+1)}
        self.volatility = {uid: initial_volatility for uid in range(0, N+1)}
        self.tau = tau

    def update_ratings(self, uids, preds, label):
        """
        Update Glicko ratings for multiple players based on individual performance outcomes.
        
        :param player_outcomes: Dictionary mapping player IDs to their outcomes (1 for win/good performance, 0 for loss/poor performance)
        :param ratings: dict mapping player IDs to their current ratings
        :param rd: dict mapping player IDs to their current rating deviations
        :param volatility: dict mapping player IDs to their current volatilities
        :param tau: system constant, smaller values (0.3 to 1.2) are more stable
        :return: updated ratings, rating deviations, and volatilities
        """

        def g(RD):
            return 1 / math.sqrt(1 + 3 * (RD**2) / (math.pi**2))

        def E(r, r_avg, RD_avg):
            return 1 / (1 + math.exp(-g(RD_avg) * (r - r_avg) / 400))

        def compute_v(g_value, E_value):
            return 1 / (g_value**2 * E_value * (1 - E_value))

        def compute_delta(v, g_value, E_value, s):
            return v * g_value * (s - E_value)

        def compute_new_volatility(sigma, delta, v, RD, tau):
            a = math.log(sigma**2)
            phi = RD
            epsilon = 0.000001

            def f(x):
                ex = math.exp(x)
                return (ex * (delta**2 - phi**2 - v - ex)) / (2 * (phi**2 + v + ex)**2) - (x - a) / (tau**2)

            A = a
            B = 0 if delta**2 <= phi**2 + v else math.log(delta**2 - phi**2 - v)

            fa = f(A)
            fb = f(B)

            while abs(B - A) > epsilon:
                C = A + (A - B) * fa / (fb - fa)
                fc = f(C)

                if fc * fb < 0:
                    A = B
                    fa = fb
                else:
                    fa = fa / 2

                B = C
                fb = fc

            return math.exp(A / 2)

        def compute_new_RD(RD, sigma):
            return math.sqrt(RD**2 + sigma**2)

        new_ratings = self.ratings.copy()
        new_rd = self.rd.copy()
        new_volatility = self.volatility.copy()

        # Calculate average rating and RD
        avg_rating = sum(self.ratings.values()) / len(self.ratings)
        avg_rd = sum(self.rd.values()) / len(self.rd)

        correct = lambda p, l: 1. if np.round(p) == l else 0
        player_outcomes = {
            uid: correct(pred, label) for uid, pred in zip(uids, preds)
        }

        for player_id, outcome in player_outcomes.items():
            r = self.ratings[player_id]
            RD = self.rd[player_id]
            sigma = self.volatility[player_id]
 
            g_value = g(avg_rd)
            E_value = E(r, avg_rating, avg_rd)

            v = compute_v(g_value, E_value)
            delta = compute_delta(v, g_value, E_value, outcome)

            new_sigma = compute_new_volatility(sigma, delta, v, RD, self.tau)
            new_RD = compute_new_RD(RD, new_sigma)
            new_RD = min(new_RD, 350)  # Cap RD at 350

            new_r = r + (new_RD**2) * g_value * (outcome - E_value)

            # Update the dictionaries
            new_ratings[player_id] = np.clip(new_r, a_min=GLICKO_MIN, a_max=GLICKO_MAX)
            new_rd[player_id] = new_RD
            new_volatility[player_id] = new_sigma

        self.ratings = new_ratings
        self.rd = new_rd
        self.volatility = new_volatility

        return [self.ratings[uid]/GLICKO_MAX for uid in uids]
