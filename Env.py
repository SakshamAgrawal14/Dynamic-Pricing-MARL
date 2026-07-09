import random


class PricingEnvironment:

    def __init__(self):
       
        # Possible prices firms can choose
        self.actions = [50,55,60,65,70]

        # Marginal cost
        self.cost = 50

        # Maximum rounds per episode
        self.max_rounds = 100

        # Initial market state
        # (previous_price_A, previous_price_B)
        self.current_state = (60, 60)

        # Track progression through episode
        self.round_number = 0


    def reset(self):

        # Reset market state
        self.current_state = (60,60)

        # Reset round counter
        self.round_number = 0

        # Return starting observation
        return self.current_state


    def step(self, actions):

        # # Extract firm prices
        price_A, price_B = actions


        # # -------------------------
        # # Demand Allocation Logic
        # # -------------------------

        # Firm A undercuts
        if price_A < price_B:

            demand_A = 1
            demand_B = 0

        # Firm B undercuts
        elif price_B < price_A:

            demand_A = 0
            demand_B = 1

        # Equal pricing
        else:

            demand_A = 0.5
            demand_B = 0.5

        # Loyal customer base
        # demand_A = 0.35
        # demand_B = 0.35


        # # Price-sensitive customers
        # price_sensitive = 0.3


        # if price_A < price_B:

        #     demand_A += price_sensitive

        # elif price_B < price_A:

        #     demand_B += price_sensitive

        # else:

        #     demand_A += price_sensitive / 2

        #     demand_B += price_sensitive / 2

        # -------------------------
        # Profit Calculation
        # π = (P - c)Q
        # -------------------------

        profit_A = (price_A - self.cost) * demand_A

        profit_B = (price_B - self.cost) * demand_B


        # -------------------------
        # State Transition
        # -------------------------

        # Current actions become next state
        self.current_state = (price_A, price_B)


        # -------------------------
        # Advance Time
        # -------------------------

        self.round_number += 1


        # -------------------------
        # Episode Termination
        # -------------------------

        done = self.round_number >= self.max_rounds


        # -------------------------
        # Return Environment Output
        # -------------------------

        return (
            self.current_state,
            (profit_A, profit_B),
            done
        )
    
