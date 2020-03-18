from math import log


class RL_trading_environment:
    def __init__(
        self,
        observation,
        next_open_variable,
        next_close_variable,
        list_to_drop,
        trade_size=10000,
        initial_balance=10000,
        spread_param=0.0005,
        transaction_cost_param=0.00002,
    ):

        """
        :param observation: it's a time series dataset containing all the observation needed to do the prediction
        :param next_open_variable: name of the column for the next open. this is used to compute the next reward
        :param next_close_variable: name of the column for the next close. this is used to compute the next reward
        :param list_to_drop: list of the features you want to drop from your original time series dataset
        :param initial_balance: initial balance
        :param spread_param: represent an estimation of the bid / ask spread in %
        :param transaction_cost_param: represent as estimation of the transaction cost in %
        """

        # define next open variable
        self.next_open = observation[next_open_variable]
        # define next close variable
        self.next_close = observation[next_close_variable]
        # training_variables
        self.observation = observation.drop(list_to_drop, axis=1)

        # set the index to 0 at beg
        self.index = 0

        # define the first observation state
        self.observation_state = self.observation.values[self.index]
        self.next_open_state = self.next_open.values[self.index]
        self.next_close_state = self.next_close.values[self.index]

        # Portfolio value
        self.current_portfolio_value = float(initial_balance)
        self.constant_trade_size = trade_size

        # spread : spread bid/ask
        self.spread = float(spread_param)

        # T_C : Transaction_Cost
        self.T_C = float(transaction_cost_param)

        # define information for the agent
        self.number_of_transactions = int(0)
        self.number_of_long_position = int(0)
        self.number_of_short_position = int(0)

        # Dictionary of states
        self.state_space = {"no_position": 0, "long_position": 1, "short_position": 2}
        # Current state of the agent
        self.current_position_state = int(0)
        # list of all the position realized
        self.position_realized = [0]

        # Dictionary of the action_space
        self.action_space = {"hold": 0, "buy": 1, "sell": 2}
        # current action taken by the the agent
        self.current_action = int(0)
        # list of all the action realized
        self.actions_realized = [0]

        # object to register the price when the agent enter a position
        self.Price_enter_Position = float(0)

        # current reward made by the agent
        self.reward = float(0)
        # list of all the reward realized
        self.cumulative_rewards = [0]

        # boolean object to determine if we reach the end of the dataset
        self.done = False

    # function to compute reward
    def reward_function(
        self,
        portfolio_value,
        entering_price,
        next_close_price,
        next_open_price,
        current_position,
        action,
    ):

        if current_position == 0 and action == 0:

            rewarded_action = 0

            new_portfolio_value = portfolio_value

            return rewarded_action, new_portfolio_value

        elif ((current_position == 0 and action == 1) or
              (current_position == 1 and action == 0) or
              (current_position == 1 & action == 1)):

            rewarded_action = log(
                (portfolio_value + min(self.constant_trade_size, portfolio_value) * (
                    (next_close_price - entering_price) / entering_price - 2 * (self.T_C + self.spread)
                ))
                / portfolio_value
            )

            new_portfolio_value = portfolio_value + min(self.constant_trade_size, portfolio_value) * (
                    (next_close_price - entering_price) / entering_price - 2 * (self.T_C + self.spread)
            )

            return rewarded_action, new_portfolio_value

        elif current_position == 1 and action == 2:

            rewarded_action = log(
                (portfolio_value + min(self.constant_trade_size, portfolio_value) * (
                        (next_open_price - entering_price) / next_open_price - 2 * (self.T_C + self.spread)
                ))
                / portfolio_value
            )

            new_portfolio_value = portfolio_value + min(self.constant_trade_size, portfolio_value) * (
                    (next_close_price - entering_price) / entering_price - 2 * (self.T_C + self.spread)
            )

            return rewarded_action, new_portfolio_value

        elif ((current_position == 0 and action == 2) or
              (current_position == 2 and action == 0) or
              (current_position == 2 and action == 2)):

            rewarded_action = log(
                (portfolio_value + min(self.constant_trade_size, portfolio_value) * (
                    (entering_price - next_close_price) / entering_price - 2 * (self.T_C + self.spread)
                ))
                / portfolio_value
            )

            new_portfolio_value = portfolio_value + min(self.constant_trade_size, portfolio_value) * (
                    (entering_price - next_close_price) / entering_price - 2 * (self.T_C + self.spread)
            )
            return rewarded_action, new_portfolio_value

        elif current_position == 2 and action == 1:

            rewarded_action = log(
                (portfolio_value + min(self.constant_trade_size, portfolio_value) * (
                    (entering_price - next_open_price) / entering_price - 2 * (self.T_C + self.spread)
                ))
                / portfolio_value
            )

            new_portfolio_value = portfolio_value + min(self.constant_trade_size, portfolio_value) * (
                    (entering_price - next_open_price) / entering_price - 2 * (self.T_C + self.spread)
            )

            return rewarded_action, new_portfolio_value

    # function to move in the environment
    def step(self, action):

        # First we register the action
        self.current_action = action
        self.actions_realized.append(self.current_action)

        # Here we define the couple [state, action] that will define the entering price position
        entering_position = [[0, 1], [0, 2]]

        if [self.current_position_state, self.current_action] in entering_position:
            # we suppose that we are able to enter in position at the next open price
            self.Price_enter_Position = self.next_open_state

            # count the number of long position
            if [self.current_position_state, self.current_action] == entering_position[0]:
                self.number_of_long_position += 1

            # count the number of short position
            elif [self.current_position_state, self.current_action] == entering_position[1]:
                self.number_of_short_position += 1

        # count the total number of transaction
        self.number_of_transactions = 2 * (self.number_of_long_position + self.number_of_short_position)

        # here we define the position our agent will be, depending on the [state, action] couple
        no_position = [[0, 0], [2, 1], [1, 2]]
        long_position = [[0, 1], [1, 1], [1, 0]]
        short_position = [[0, 2], [2, 2], [2, 0]]

        # now we define the position of the agent
        if [self.current_position_state, self.current_action] in no_position:
            self.current_position_state = int(0)

        elif [self.current_position_state, self.current_action] in long_position:
            self.current_position_state = int(1)

        elif [self.current_position_state, self.current_action] in short_position:
            self.current_position_state = int(2)

        self.position_realized.append(self.current_position_state)

        # base on the new action (and the sequence of the previous actions) we compute the reward
        self.reward, self.current_portfolio_value = self.reward_function(
            self.current_portfolio_value,
            self.Price_enter_Position,
            self.next_close_state,
            self.next_open_state,
            self.current_position_state,
            self.current_action,
        )

        self.cumulative_rewards.append(self.reward)

        # after we update the reward, we
        self.index += 1

        # update the observations, next_open and next_close
        self.observation_state = self.observation.values[self.index]
        self.next_open_state = self.next_open.values[self.index]
        self.next_close_state = self.next_close.values[self.index]

        # finally, we want to reset the index each time that we achieve the end of the dataset
        self.done = self.done_function(self.index, len(self.observation))

        if self.done:
            self.index = 0

        # this function output 3 elements : the next observations {S(t+1)}, the reward {R(t)} and if we
        return self.observation_state, self.reward, self.done

    def done_function(self, index_position, len_dataset):
        return index_position == (len_dataset-1)


    def get_agent_current_status(self):

        print("Current Agent Position ", self.current_position_state)
        print('------------------------')
        print("Last Action of the Agent ", self.current_action)
        print('------------------------')
        print("Current Portfolio Value ", self.current_portfolio_value)
        print('------------------------')
        print(" Last Reward ", self.reward)

        print('------------------------')
        print("number of transaction ", self.number_of_transactions)
        print("number of short position ", self.number_of_short_position)
        print("number of long position ", self.number_of_long_position)

    def reset(self):

        self.index = 0

        self.observation_state = self.observation.values[self.index]
        self.next_open_state = self.next_open.values[self.index]
        self.next_close_state = self.next_close.values[self.index]

        self.done = False

        return self.observation_state
