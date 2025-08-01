import random
import numpy as np
import collections
import math
import price_models as pm
from rewards import RewardFunction
from actions import action_registry
# ------------------------------------------------ Financial Parameters --------------------------------------------------- #

ANNUAL_VOLAT = 0.12                                # Annual volatility in stock price
BID_ASK_SP = 1 / 8                                 # Bid-ask spread
DAILY_TRADE_VOL = 5e6                              # Average Daily trading volume  
TRAD_DAYS = 250                                    # Number of trading days in a year
DAILY_VOLAT = ANNUAL_VOLAT / np.sqrt(TRAD_DAYS)    # Daily volatility in stock price

# ----------------------------- Parameters for the Almgren and Chriss Optimal Execution Model ----------------------------- #

TOTAL_SHARES = 1000000                                               # Total number of shares to sell
STARTING_PRICE = 50                                                  # Starting price per share
LLAMBDA = 1e-6                                                       # Trader's risk aversion
LIQUIDATION_TIME = 60                                                # How many days to sell all the shares. 
NUM_N = 60                                                           # Number of trades
EPSILON = BID_ASK_SP / 2                                             # Fixed Cost of Selling.
SINGLE_STEP_VARIANCE = (DAILY_VOLAT  * STARTING_PRICE) ** 2          # Calculate single step variance
ETA = BID_ASK_SP / (0.01 * DAILY_TRADE_VOL)                          # Price Impact for Each 1% of Daily Volume Traded
GAMMA = BID_ASK_SP / (0.1 * DAILY_TRADE_VOL)                         # Permanent Impact Constant
# COMMISSION_RATE = 0.01
# ----------------------------------------------------------------------------------------------------------------------- #


# Simulation Environment

class MarketEnvironment():
    
    def __init__(self, randomSeed = 0,
                 lqd_time = LIQUIDATION_TIME,
                 num_tr = NUM_N,
                 lambd = LLAMBDA,
                 mu = 0.0,
                 alpha: float = 2.0,
                 leftover_penalty: float = 1e-3,
                 reward_function: RewardFunction = None,
                 price_model: str = 'gbm'):
        
        # Set the random seed
        random.seed(randomSeed)
        self.price_model = price_model
        self.reward_function = reward_function

        # Initialize the financial parameters so we can access them later
        self.anv = ANNUAL_VOLAT
        self.basp = BID_ASK_SP
        self.dtv = DAILY_TRADE_VOL
        self.sigma = DAILY_VOLAT
        self.dpv = DAILY_VOLAT
        self.mu = mu
        self.alpha = alpha
        self.leftover_penalty = leftover_penalty

        # Initialize the Almgren-Chriss parameters so we can access them later
        self.total_shares = TOTAL_SHARES
        self.startingPrice = STARTING_PRICE
        self.llambda = lambd
        self.liquidation_time = lqd_time
        self.num_n = num_tr
        self.epsilon = EPSILON
        self.singleStepVariance = SINGLE_STEP_VARIANCE
        self.eta = ETA
        self.gamma = GAMMA
        
        # Calculate some Almgren-Chriss parameters
        self.tau = self.liquidation_time / self.num_n 
        self.eta_hat = self.eta - (0.5 * self.gamma * self.tau)
        self.kappa_hat = np.sqrt((self.llambda * self.singleStepVariance) / self.eta_hat)
        self.kappa = np.arccosh((((self.kappa_hat ** 2) * (self.tau ** 2)) / 2) + 1) / self.tau
        self.sigma = ANNUAL_VOLAT
        self.dt = self.tau / TRAD_DAYS

        # Set the variables for the initial state
        self.shares_remaining = self.total_shares
        self.timeHorizon = self.num_n
        self.logReturns = collections.deque(np.zeros(6))
        
        # Set the initial impacted price to the starting price
        self.prevImpactedPrice = self.startingPrice

        # Set the initial transaction state to False
        self.transacting = False
        
        # Set a variable to keep trak of the trade number
        self.k = 0

        self.lagCoeffs = np.array([1.5, -0.9, 0.8, -0.6, 0.5, -0.3])     
        self.alphas = np.array([1, 0.4, 0.3, 0.2, 0.1, 0.1]) * 30
        self.a_deque = collections.deque(np.zeros(len(self.alphas)))
        self.returns_deque = collections.deque(np.zeros(len(self.lagCoeffs)))                
        self.unaffected_returns_deque = collections.deque(np.zeros(len(self.lagCoeffs)))   
        self.logReturns = collections.deque(np.zeros(len(self.lagCoeffs)))
        # Constant multiplier for the action returned by the Actor-Critic Model
        self.constantSharesToSell = (self.total_shares / self.num_n) * 12

        
        
    def reset(self, seed = 0, liquid_time = LIQUIDATION_TIME, num_trades = NUM_N, lamb = LLAMBDA):
        
        # Initialize the environment with the given parameters
        self.__init__(randomSeed = seed, lqd_time = liquid_time, num_tr = num_trades, lambd = lamb,reward_function=self.reward_function)
        
        # Set the initial state to [0,0,0,0,0,0,1,1]
        self.initial_state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n, \
                                                               self.shares_remaining / self.total_shares])
        if self.reward_function is not None:
            self.reward_function.reset(self.initial_state)

        return self.initial_state

    
    def start_transactions(self):
        
        # Set transactions on
        self.transacting = True
        
        # Set the minimum number of stocks one can sell
        self.tolerance = 1
        
        # Set the initial capture to zero
        self.totalCapture = 0
        
        # Set the initial previous price to the starting price
        self.prevPrice = self.startingPrice
        self.P0 = self.prevPrice

        # Set the initial square of the shares to sell to zero
        self.totalSSSQ = 0
        
        # Set the initial square of the remaing shares to sell to zero
        self.totalSRSQ = 0
        
        # Set the initial AC utility
        self.prevUtility = self.compute_AC_utility(self.total_shares)
        

    def step(self, action,action_type):
        
        # Create a class that will be used to keep track of information about the transaction
        class Info(object):
            pass        
        info = Info()
        
        # Set the done flag to False. This indicates that we haven't sold all the shares yet.
        info.done = False
        
        current_state = np.array(
            list(self.logReturns) +
            [self.timeHorizon / self.num_n,
            self.shares_remaining / self.total_shares]
        )

        # During training, if the DDPG fails to sell all the stocks before the given 
        # number of trades or if the total number shares remaining is less than 1, then stop transacting,
        # set the done Flag to True, return the current implementation shortfall, and give a negative reward.
        # The negative reward is given in the else statement below.
        if self.transacting and (self.timeHorizon == 0 or abs(self.shares_remaining) < self.tolerance):
            self.transacting = False
            info.done = True
            info.implementation_shortfall = self.total_shares * self.startingPrice - self.totalCapture
            info.expected_shortfall = self.get_expected_shortfall(self.total_shares)
            info.expected_variance = self.singleStepVariance * self.tau * self.totalSRSQ
            info.utility = info.expected_shortfall + self.llambda * info.expected_variance
            
        # We don't add noise before the first trade    
        if self.k == 0:
            info.price = self.prevImpactedPrice
            self.k += 1
        else:
            if self.price_model == 'abm':
                info.price = pm.abm_price(self.prevImpactedPrice, self.singleStepVariance, self.tau)
            elif self.price_model == 'ar_l':
                info.price = pm.ar_l_price(self.prevImpactedPrice, self.alphas,
                                        self.lagCoeffs, self.a_deque, self.returns_deque)
            elif self.price_model == 'gbm':
                info.price = pm.gbm_price(
                    prev_price=self.prevImpactedPrice,
                    dt=self.dt,
                    sigma=self.sigma,
                    mu=self.mu
                )
                self.prevImpactedPrice = info.price

        # If we are transacting, the stock price is affected by the number of shares we sell. The price evolves 
        # according to the Almgren and Chriss price dynamics model. 
        if self.transacting:
            
            # If action is an ndarray then extract the number from the array
            if isinstance(action, np.ndarray):
                action = action.item()            

            # Use modular strategy class from action registry
            if action_type not in action_registry:
                raise ValueError(f"Unknown action_type '{action_type}'")

            sharesToSellNow = action_registry[action_type].compute(self, action)

        
            if self.timeHorizon < 2:
                sharesToSellNow = self.shares_remaining

            # Since we are not selling fractions of shares, round up the total number of shares to sell to the nearest integer. 
            info.share_to_sell_now = np.around(sharesToSellNow)

            # Calculate the permanent and temporary impact on the stock price according the AC price dynamics model
            info.currentPermanentImpact = self.permanentImpact(info.share_to_sell_now)
            info.currentTemporaryImpact = self.temporaryImpact(info.share_to_sell_now)
                
            # Apply the temporary impact on the current stock price    
            info.exec_price = info.price - info.currentTemporaryImpact
            best_limit_price = info.price - (self.basp / 2)         # mid‑spread assumption
            info.trade_through = (info.exec_price < best_limit_price)
            
            # Calculate the current total capture
            self.totalCapture += info.share_to_sell_now * info.exec_price

            # Calculate the log return for the current step and save it in the logReturn deque
            self.logReturns.append(np.log(info.price/self.prevPrice))
            self.logReturns.popleft()
            
            # Update the number of shares remaining
            self.shares_remaining -= info.share_to_sell_now
            
            # Calculate the runnig total of the squares of shares sold and shares remaining
            self.totalSSSQ += info.share_to_sell_now ** 2
            self.totalSRSQ += self.shares_remaining ** 2
                                        
            # Update the variables required for the next step
            self.timeHorizon -= 1
            self.prevPrice = info.price
            self.prevImpactedPrice = info.price - info.currentPermanentImpact
            
            # Calculate the reward
            # currentUtility = self.compute_AC_utility(self.shares_remaining)
            # reward = (abs(self.prevUtility) - abs(currentUtility)) / abs(self.prevUtility)
            # self.prevUtility = currentUtility

            # ---------------------------------------------------------------------------------
            # # Q_t = shares actually sold this step
            # Q_t = info.share_to_sell_now
            # # P_t = average executed price
            # P_t = info.exec_price
            # #       since temporaryImpact = ε·sign(Q)+η/τ·Q, one can invert η/τ to get 'levels'
            # d_t = abs(info.currentTemporaryImpact) / (self.eta / self.tau)
            # reward = Q_t * (self.P0 - P_t) - self.alpha * d_t
            # # if you want a hard end‐penalty for leftover shares:
            # if info.done and self.shares_remaining>0:
            #     reward -= self.leftover_penalty * self.shares_remaining
            # # convex block-size penalty
            # beta = 1e-6
            # reward -= beta * (Q_t**2)
            # # per-step inventory cost
            # delta = 1e-3
            # reward -= delta * (self.shares_remaining / self.total_shares)
            # frac = Q_t/self.total_shares
            # reward -= 1e5 * (np.exp(5*frac) - 1)
            # ----------------------------------------------------------------------------------

            # Q_t   = info.share_to_sell_now 
            # r_bar = Q_t * info.exec_price 
            # # normalized reward:
            # reward = (r_bar - Q_t * self.P0) / self.total_shares
            # ----------------------------------------------------------------------------------

            if self.reward_function is not None:
                next_state = np.array(
                    list(self.logReturns) +
                    [self.timeHorizon / self.num_n,
                    self.shares_remaining / self.total_shares]
                )
                curr   = current_state.reshape(1, -1)
                act    = np.array([action]).reshape(1, -1)
                nxt    = next_state.reshape(1, -1)
                reward = float(self.reward_function.calculate(
                    curr, act, nxt, info.done
                ))
            # If all the shares have been sold calculate E, V, and U, and give a positive reward.
            if self.shares_remaining <= 0:
                
                # Calculate the implementation shortfall
                info.implementation_shortfall  = self.total_shares * self.startingPrice - self.totalCapture

                # Set the done flag to True. This indicates that we have sold all the shares
                info.done = True
        else:
            reward = 0.0

        self.k += 1
            
        # Set the new state
        state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n, self.shares_remaining / self.total_shares])

        return (state, np.array([reward]), info.done, info)

    def permanentImpact(self, sharesToSell):
        # Calculate the permanent impact according to equations (6) and (1) of the AC paper
        pi = self.gamma * sharesToSell
        return pi

    
    def temporaryImpact(self, sharesToSell):
        # Calculate the temporary impact according to equation (7) of the AC paper
        ti = (self.epsilon * np.sign(sharesToSell)) + ((self.eta / self.tau) * sharesToSell) \
            # + COMMISSION_RATE * sharesToSell * np.sign(sharesToSell)
        return ti
    
    def get_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall according to equation (8) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)        
        st = self.epsilon * sharesToSell
        tt = (self.eta_hat / self.tau) * self.totalSSSQ
        return ft + st + tt

    
    def get_AC_expected_shortfall(self, sharesToSell):
        # Calculate the expected shortfall for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)        
        st = self.epsilon * sharesToSell        
        tt = self.eta_hat * (sharesToSell ** 2)       
        nft = np.tanh(0.5 * self.kappa * self.tau) * (self.tau * np.sinh(2 * self.kappa * self.liquidation_time) \
                                                      + 2 * self.liquidation_time * np.sinh(self.kappa * self.tau))       
        dft = 2 * (self.tau ** 2) * (np.sinh(self.kappa * self.liquidation_time) ** 2)   
        fot = nft / dft       
        return ft + st + (tt * fot)  
        
    
    def get_AC_variance(self, sharesToSell):
        # Calculate the variance for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * (self.singleStepVariance) * (sharesToSell ** 2)                        
        nst  = self.tau * np.sinh(self.kappa * self.liquidation_time) * np.cosh(self.kappa * (self.liquidation_time - self.tau)) \
               - self.liquidation_time * np.sinh(self.kappa * self.tau)        
        dst = (np.sinh(self.kappa * self.liquidation_time) ** 2) * np.sinh(self.kappa * self.tau)        
        st = nst / dst
        return ft * st
        
        
    def compute_AC_utility(self, sharesToSell):    
        # Calculate the AC Utility according to pg. 13 of the AC paper
        if self.liquidation_time == 0:
            return 0        
        E = self.get_AC_expected_shortfall(sharesToSell)
        V = self.get_AC_variance(sharesToSell)
        return E + self.llambda * V
    
    
    def get_trade_list(self):
        # Calculate the trade list for the optimal strategy according to equation (18) of the AC paper
        trade_list = np.zeros(self.num_n)
        ftn = 2 * np.sinh(0.5 * self.kappa * self.tau)
        ftd = np.sinh(self.kappa * self.liquidation_time)
        ft = (ftn / ftd) * self.total_shares
        for i in range(1, self.num_n + 1):       
            st = np.cosh(self.kappa * (self.liquidation_time - (i - 0.5) * self.tau))
            trade_list[i - 1] = st
        trade_list *= ft
        return trade_list

        
    def observation_space_dimension(self):
        # Return the dimension of the state
        return 8
    
    
    def action_space_dimension(self):
        # Return the dimension of the action
        return 1
    
    
    def stop_transactions(self):
        # Stop transacting
        self.transacting = False            
            