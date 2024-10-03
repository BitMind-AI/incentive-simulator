import math

#class GlickoRater

def update_glicko_ratings(player_outcomes, ratings, rd, volatility, tau=0.5):
    """
    Update Glicko ratings for multiple players based on individual performance outcomes.
    
    :param player_outcomes: Dictionary mapping player IDs to their outcomes (1 for win/good performance, 0 for loss/poor performance)
    :param ratings: dict mapping player IDs to their current ratings
    :param rd: dict mapping player IDs to their current rating deviations
    :param volatility: dict mapping player IDs to their current volatilities
    :param tau: system constant, smaller values (0.3 to 1.2) are more stable
    :return: updated ratings, rating deviations, and volatilities

    # Example usage:
    ratings = {1: 1500, 2: 1400, 3: 1550, 4: 1700}
    rd = {1: 200, 2: 30, 3: 100, 4: 300}
    volatility = {1: 0.06, 2: 0.06, 3: 0.06, 4: 0.06}
    
    # Simulating a round where each player either performs well (1) or poorly (0)
    player_outcomes = {
        1: 1,  # Player 1 performed well
        2: 0,  # Player 2 performed poorly
        3: 1,  # Player 3 performed well
        4: 0,  # Player 4 performed poorly
    }
    
    new_ratings, new_rd, new_volatility = update_glicko_ratings(player_outcomes, ratings, rd, volatility)
    print(f"Updated ratings: {new_ratings}")
    print(f"Updated RDs: {new_rd}")
    print(f"Updated volatilities: {new_volatility}")

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

    new_ratings = ratings.copy()
    new_rd = rd.copy()
    new_volatility = volatility.copy()

    # Calculate average rating and RD
    avg_rating = sum(ratings.values()) / len(ratings)
    avg_rd = sum(rd.values()) / len(rd)

    for player_id, outcome in player_outcomes.items():
        r = ratings[player_id]
        RD = rd[player_id]
        sigma = volatility[player_id]
        
        g_value = g(avg_rd)
        E_value = E(r, avg_rating, avg_rd)
        
        v = compute_v(g_value, E_value)
        delta = compute_delta(v, g_value, E_value, outcome)
        
        new_sigma = compute_new_volatility(sigma, delta, v, RD, tau)
        new_RD = compute_new_RD(RD, new_sigma)
        new_RD = min(new_RD, 350)  # Cap RD at 350
        
        new_r = r + (new_RD**2) * g_value * (outcome - E_value)
        
        # Update the dictionaries
        new_ratings[player_id] = new_r
        new_rd[player_id] = new_RD
        new_volatility[player_id] = new_sigma
    
    return new_ratings, new_rd, new_volatility
