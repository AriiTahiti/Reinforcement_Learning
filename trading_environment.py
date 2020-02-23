from math import log


class RL_trading_environment:
    def __init__(
        self,
        observation,
        next_open_variable,
        next_close_variable,
        list_to_drop,
        initial_balance=100000,
        spread_param=0.0005,
        transaction_cost_param=0.00002,
    ):

        """
        :param observation: it's a time series dataset containing all the observation needed to do the prediction
        :param next_open_variable: name of the column for the next open. this is used to compute the next reward
        :param next_close_variable: name of the column for the next close. this is used to compute the next reward
        :param list_to_drop: list of the features you want to drop from your original time series dataset
        :param initial_balance: initial balance
        :param spread_param:
        :param transaction_cost_param:
        """

        #
        self.next_open = observation[next_open_variable]
        self.next_close = observation[next_close_variable]
        self.observation = observation.drop(list_to_drop, axis=1)

        self.index = 0

        self.observation_state = self.observation.values[self.index]
        self.next_open_state = self.next_open.values[self.index]
        self.next_close_state = self.next_close.values[self.index]

        # Portfolio
        self.current_portfolio_value = float(initial_balance)

        # spread : spread bid/ask
        self.spread = float(spread_param)

        # T_C : Transaction_Cost
        self.T_C = float(transaction_cost_param)

        # Agent parameters : This part is to save information about the learning experience of the agent
        self.number_of_transactions = int(0)
        self.number_of_long_position = int(0)
        self.number_of_short_position = int(0)

        # Dictionary of states
        self.state_space = {"no_position": 0, "long_position": 1, "short_position": 2}
        self.current_position_state = int(0)
        self.position_realized = [0]

        # action_space is used to map the action number with the definition
        self.action_space = {"hold": 0, "buy": 1, "sell": 2}
        self.current_action = int(0)
        self.actions_realized = [0]

        # P_en_P : Price_enter_Position
        self.Price_enter_Position = float(0)

        # rewards cumulative
        self.reward = float(0)
        self.cumulative_rewards = float(0)

        # done
        self.done = False


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
            return rewarded_action

        elif (
            (current_position == 0 and action == 1)
            or (current_position == 1 and action == 0)
            or (current_position == 1 & action == 1)
        ):

            rewarded_action = log(
                portfolio_value
                * (
                    1
                    + (next_close_price - entering_price) / entering_price
                    - 2 * (self.T_C + self.spread)
                )
                / portfolio_value
            )
            return rewarded_action

        elif current_position == 1 and action == 2:
            rewarded_action = log(
                portfolio_value
                * (
                    1
                    + (next_open_price - entering_price) / next_open_price
                    - 2 * (self.T_C + self.spread)
                )
                / portfolio_value
            )
            return rewarded_action

        elif (
            (current_position == 0 and action == 2)
            or (current_position == 2 and action == 0)
            or (current_position == 2 and action == 2)
        ):
            rewarded_action = log(
                portfolio_value
                * (
                    1
                    + (entering_price - next_close_price) / entering_price
                    - 2 * (self.T_C + self.spread)
                )
                / portfolio_value
            )
            return rewarded_action

        elif current_position == 2 and action == 1:
            rewarded_action = log(
                portfolio_value
                * (
                    1
                    + (entering_price - next_open_price) / entering_price
                    - 2 * (self.T_C + self.spread)
                )
                / portfolio_value
            )
            return rewarded_action

        else:
            return print("reward has bug")

    """
    step function is used to make the transition between the current state and the next state depending on the action
    """

    def step(self, action):

        # First we register the action
        self.current_action = action
        self.actions_realized.append(self.current_action)

        # Here we define the couple [state, action] that will define the entering price position
        entering_position = [[0, 1], [0, 2]]

        if [self.current_position_state, self.current_action] in entering_position:
            self.Price_enter_Position = self.next_open_state

            # count the number of long position
            if [self.current_position_state, self.current_action] == entering_position[
                0
            ]:
                self.number_of_long_position += 1

            # count the number of short position
            elif [
                self.current_position_state,
                self.current_action,
            ] == entering_position[0]:
                self.number_of_short_position += 1

        # count the total number of transaction
        self.number_of_transactions = 2 * (
            self.number_of_long_position + self.number_of_short_position
        )

        # here we define the state our agent will be depending on the [state, action] couple
        position_0 = [[0, 0], [2, 1], [1, 2]]
        position_1 = [[0, 1], [1, 1], [1, 0]]
        position_2 = [[0, 2], [2, 2], [2, 0]]

        if [self.current_position_state, self.current_action] in position_0:
            self.current_position_state = int(0)

        elif [self.current_position_state, self.current_action] in position_1:
            self.current_position_state = int(1)

        elif [self.current_position_state, self.current_action] in position_2:
            self.current_position_state = int(2)

        self.position_realized.append(self.current_position_state)

        # base on the new action (and the sequence of the previous actions) we compute the reward
        self.reward = self.reward_function(
            self.current_portfolio_value,
            self.Price_enter_Position,
            self.next_close_state,
            self.next_open_state,
            self.current_position_state,
            self.current_action,
        )

        self.cumulative_rewards = self.current_portfolio_value * self.reward

        # after we update the reward, we
        self.index += 1

        # update the observations, next_open and next_close
        self.observation_state = self.observation.values[self.index]
        self.next_open_state = self.next_open.values[self.index]
        self.next_close_state = self.next_close.values[self.index]

        # finally, we want to reset the index each time that we achieve the end of the dataset
        self.done = self.done_function(self.index, (len(self.observation) - 1))

        if self.done:
            self.index = 0

        return self.observation_state, self.reward, self.done

    def done_function(self, index_position, lenght_dataset):
        return index_position == lenght_dataset

    def get_agent_current_status(self):

        print("Current Agent Position ", self.current_position_state)
        print("Last Action of the Agent ", self.current_action)
        print("Current Balance ", self.current_portfolio_value)

        print("number of transaction ", self.number_of_transactions)
        print("number of short position ", self.number_of_short_position)
        print("number of long position ", self.number_of_long_position)

        print("price enter position ", self.Price_enter_Position)
        print("next_open_state ", self.next_open_state)
        print("next_close_state ", self.next_close_state)

        print("reward ", self.reward)

    def reset(self):

        self.index = 0

        self.observation_state = self.observation.values[self.index]
        self.next_open_state = self.next_open.values[self.index]
        self.next_close_state = self.next_close.values[self.index]

        self.current_position_state = 0
        self.current_action = 0

        # P_en_P : Price_enter_Position
        self.P_en_P = 0

        # rewards cumulative
        self.cumulative_rewards = 0

        # done
        self.done = False

        return self.observation_state
