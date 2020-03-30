from math import log


class RLTradingEnvironment:

    def __init__(
        self,
        observation,
        next_open_variable,
        next_close_variable,
        list_to_drop,
        trade_size=10000,
        initial_balance=10000,
        spread_param=0.0005,
        transaction_cost=0.00002,
    ):

        """
        This function is used to instantiate the class. This include different variables that will be used to
        analyse the performance of the agent in this specific environment

        Args:
            observation: it's a time series dataset containing all the observation needed to do the prediction
            next_open_variable: name of the column for the next open. this is used to compute the next reward
            next_close_variable: name of the column for the next close. this is used to compute the next reward
            list_to_drop: list of the features you want to drop from your original time series dataset
            trade_size: maximum fixed trade size
            initial_balance: initial balance
            spread_param: represent an estimation of the bid / ask spread in %
            transaction_cost: represent as estimation of the transaction cost in %
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

        # Portfolio Value
        self.original_portfolio_value = float(initial_balance)
        self.current_portfolio_value = float(initial_balance)
        self.maximum_trade_size = trade_size
        self.traded_portfolio_value = min(trade_size, initial_balance)

        # total_transaction_cost is the sum of bid/ask spread and transaction cost
        self.total_transaction_cost = float(spread_param) + float(transaction_cost)

        # define information for the agent
        self.number_of_transactions = int(0)
        self.number_of_long_position = int(0)
        self.number_of_short_position = int(0)

        # Dictionary of position
        self.state_space = {"no_position": 0, "long_position": 1, "short_position": 2}
        # position of the agent
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
        self.price_enter_position = float(0)
        # object to register the last traded amount
        self.last_traded_amount = float(0)

        # current reward made by the agent
        self.reward = float(0)
        # list of all the reward realized
        self.all_step_rewards = [0]
        # portfolio log return
        self.log_return_portfolio = 0


        # boolean object to determine if we reach the end of the dataset
        self.done = False

    def reward_function(
        self,
        traded_amount: float,
        current_portfolio_amount: float,
        entering_price: float,
        next_close_price: float,
        next_open_price: float,
        current_position: int,
        action: int,
    ):

        """
        Args:
            traded_amount: is the amount in the current position open
            current_portfolio_amount : it'sthe current portfolio amount
            entering_price: price we enter into the position
            next_close_price: it's the next close price to compute the reward if we stay in current position
            next_open_price:  it's the next open price to compute the reward if we leave the current position
            current_position: it's the current position state of the agent
            action: is the current action taken by the agent

        Returns: the function output the return amount on the trading decision.

        """
        if current_position == 0 and action == 0:
            rewarded_action = 0

        elif (
            (current_position == 0 and action == 1)
            or (current_position == 1 and action == 0)
            or (current_position == 1 & action == 1)
        ):

            rewarded_action = traded_amount * (
                (next_close_price - entering_price) / entering_price
                - 2 * self.total_transaction_cost
            )

        elif current_position == 1 and action == 2:

            rewarded_action = traded_amount * (
                (next_open_price - entering_price) / next_open_price
                - 2 * self.total_transaction_cost
            )

        elif (
            (current_position == 0 and action == 2)
            or (current_position == 2 and action == 0)
            or (current_position == 2 and action == 2)
        ):

            rewarded_action = traded_amount * (
                (entering_price - next_close_price) / entering_price
                - 2 * self.total_transaction_cost
            )

        elif current_position == 2 and action == 1:

            rewarded_action = traded_amount * (
                (entering_price - next_open_price) / entering_price
                - 2 * self.total_transaction_cost
            )

        else:
            raise ValueError("current action or current position is not well defined")

        new_portfolio_value = current_portfolio_amount + rewarded_action

        return rewarded_action, new_portfolio_value

    # function to move in the environment
    def step(self, action: int):
        """
        This function make a move in the time series dataset, and update the situation of the agent in the environment


        Args:
            action: Take the action decided by the agent

        Returns: This function returns 3 elements.
            The new observation available for the agent
            The reward realised by the agent (the one used to train the model)
            The indicator that determine if you reached the end of the dataset

        """

        # if done than reset data :
        if self.done:
            print("--- Resetting Time Series ---")
            self.reset()

        # First we register the action
        self.current_action = action
        self.actions_realized.append(self.current_action)

        # Here we define the couple [state, action] that will define the entering price position
        entering_position = [[0, 1], [0, 2]]

        if [self.current_position_state, self.current_action] in entering_position:
            # we suppose that we are able to enter in position at the next open price
            self.price_enter_position = self.next_open_state
            self.last_traded_amount = min(
                self.current_portfolio_value, self.maximum_trade_size
            )

            # count the number of long position
            if [self.current_position_state, self.current_action] == entering_position[
                0
            ]:
                self.number_of_long_position += 1

            # count the number of short position
            elif [
                self.current_position_state,
                self.current_action,
            ] == entering_position[1]:
                self.number_of_short_position += 1

        # count the total number of transaction
        self.number_of_transactions = 2 * (
            self.number_of_long_position + self.number_of_short_position
        )

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

        (
            self.reward,
            self.traded_portfolio_value,
        ) = self.reward_function(
            self.last_traded_amount,
            self.current_portfolio_value,
            self.price_enter_position,
            self.next_close_state,
            self.next_open_state,
            self.current_position_state,
            self.current_action,
        )

        self.all_step_rewards.append(self.reward)

        self.log_return_portfolio = 0

        # update current portfolio_value [state, action] :
        exit_position = [[1, 2], [2, 1]]
        if [self.current_position_state, self.current_action] in exit_position:
            self.log_return_portfolio = log(self.traded_portfolio_value/self.current_portfolio_value)
            self.current_portfolio_value = self.traded_portfolio_value

        # after we update the reward
        self.index += 1

        # finally, we want to reset the index each time that we achieve the end of the dataset
        if self.index == (len(self.observation) - 1):
            self.done = True

        # update the observations, next_open and next_close
        self.observation_state = self.observation.values[self.index]
        self.next_open_state = self.next_open.values[self.index]
        self.next_close_state = self.next_close.values[self.index]

        # this function output 3 elements : the next observations {S(t+1)}, the reward {R(t)} and if we
        return self.observation_state, self.log_return_portfolio, self.done

    def get_agent_current_status(self):
        """
        Returns: This function only print the current information about the environment
        """

        print("Current Agent Position ", self.current_position_state)
        print("------------------------")
        print("Last Action of the Agent ", self.current_action)
        print("------------------------")
        print("Current Portfolio Value ", self.current_portfolio_value)
        print("------------------------")
        print(" Last Reward ", self.reward)

        print("------------------------")
        print("number of transaction ", self.number_of_transactions)
        print("number of short position ", self.number_of_short_position)
        print("number of long position ", self.number_of_long_position)

    def reset(self):
        """
        Returns: The reset function is used to rest the time series, so the agent will start to learn again on
        the same dataset
        """
        self.index = 0

        self.observation_state = self.observation.values[self.index]
        self.next_open_state = self.next_open.values[self.index]
        self.next_close_state = self.next_close.values[self.index]

        self.done = False

        self.current_portfolio_value = self.original_portfolio_value

        # define information for the agent
        self.number_of_transactions = int(0)
        self.number_of_long_position = int(0)
        self.number_of_short_position = int(0)

        # current reward made by the agent
        self.reward = float(0)
        # list of all the reward realized
        self.cumulative_rewards = [0]
        # portfolio log return
        self.log_return_portfolio = 0

        # object to register the price when the agent enter a position
        self.price_enter_position = float(0)
        # object to register the last traded amount
        self.last_traded_amount = float(0)

        # position of the agent
        self.current_position_state = int(0)
        # list of all the position realized
        self.position_realized = [0]

        # current action taken by the the agent
        self.current_action = int(0)
        # list of all the action realized
        self.actions_realized = [0]
