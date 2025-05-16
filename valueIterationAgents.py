# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

# Question 1: Value Iteration
# Test commands:
# python autograder.py -q q1
# python gridworld.py -a value -i 100 -k 10
# python gridworld.py -a value -i 5
class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations # Number of iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # todo: ValueIterationAgent.runValueIteration()

        """
        Run value iteration for a fixed number of iterations
        Bellman Optimality Equation:
        V(s) ← max_a Σ T(s,a,s') * [R(s,a,s') + γ * V(s')]
        """
        for _ in range(self.iterations):
            updated_values = util.Counter()  # New value table for this iteration

            for state in self.mdp.getStates():  # Iterate over all states
                if self.mdp.isTerminal(state):
                    # If it is a termination state, the value is set to 0.
                    updated_values[state] = 0.0
                    continue

                # Compute Q-values for all actions and take the maximum as V(s)
                max_q_value = max(
                    self.computeQValueFromValues(state, action)
                    for action in self.mdp.getPossibleActions(state)
                )

                updated_values[state] = max_q_value

            self.values = updated_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

          Bellman Optimality Equation:
          V(s) ← max_a Σ T(s,a,s') * [R(s,a,s') + γ * V(s')]

          Q(s,a) = Σ T(s,a,s') * [R(s,a,s') + γ * V(s')]
        """

        q_value = 0.0
        for next_state, pr in self.mdp.getTransitionStatesAndProbs(state, action):
            # R(s,a,s')
            reward = self.mdp.getReward(state, action, next_state)

            # T(s,a,s') * [R(s,a,s') + γ * V(s')]
            q_value += pr * (reward + self.discount * self.values[next_state])

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # todo: ValueIterationAgent.computeActionFromValues()
        # Return None if this is a terminal state
        if self.mdp.isTerminal(state):
            return None

        # Get all valid actions in the state
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None

        # π(s) = argmax_a Q(s,a)
        # Pick the action with the highest Q-value
        best_action = max(
            actions,
            key=lambda action: self.computeQValueFromValues(state, action)
        )

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

# Question 4: Asynchronous Value Iteration
# test:
#   python autograder.py -q q4
#   python gridworld.py -a asynchvalue -i 1000 -k 10
class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # todo: runValueIteration
        """
        The reason this class is called AsynchronousValueIterationAgent
        because we will update only one state in each iteration,
        as opposed to doing a batch-style update.

        Here is how cyclic value iteration works. In the first iteration,
        only update the value of the first state in the states list. In
        the second iteration, only update the value of the second.
        Keep going until you have updated the value of each state once,
        then start back at the first state for the subsequent iteration.

        If the state picked for updating is terminal, nothing happens in that iteration.
        You can implement it as indexing into the states variable defined in the code skeleton.
        """
        states = self.mdp.getStates()

        for k in range(self.iterations):
            # Pick one state in order each time
            curr_state = states[k % len(states)]

            # Skip terminal state
            if self.mdp.isTerminal(curr_state):
                continue

            actions = self.mdp.getPossibleActions(curr_state)
            if not actions:
                self.values[curr_state] = 0.0
                continue

            # Compute Q(s,a) for each action and take max as V(s)
            best_q = max(
                self.computeQValueFromValues(curr_state, action)
                for action in actions
            )

            # Update value of the current state
            self.values[curr_state] = best_q

# Question 5: Prioritized Sweeping Value Iteration
# test:
#   python autograder.py -q q5
#   python gridworld.py -a priosweepvalue -i 1000
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        1. Compute predecessors of all states.
        2. Initialize an empty priority queue.
        3. For each non-terminal state s, do: (note: to make the autograder work for this question, you must iterate over
            states in the order returned by self.mdp.getStates())
            o Find the absolute value of the difference between the current value of s in self.values and the highest
                Q-value across all possible actions from s (this represents what the value should be); call this number
                diff. Do NOT update self.values[s] in this step.
            o Push s into the priority queue with priority -diff (note that this is negative). We use a negative because
                the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
        4. For iteration in 0, 1, 2, ..., self.iterations - 1, do:
            o If the priority queue is empty, then terminate.
            o Pop a state s off the priority queue.
            o Update s's value (if it is not a terminal state) in self.values.
            o For each predecessor p of s, do:
                - Find the absolute value of the difference between the current value of p in self.values and the
                    highest Q-value across all possible actions from p (this represents what the value should be);
                    call this number diff. Do NOT update self.values[p] in this step.
                - If diff > theta, push p into the priority queue with priority -diff (note that this is negative),
                    as long as it does not already exist in the priority queue with equal or lower priority.
                    As before, we use a negative because the priority queue is a min heap, but we want to prioritize
                    updating states that have a higher error.

        """
        "*** YOUR CODE HERE ***"
        # 1. Compute predecessors of all states.
        predecessors = collections.defaultdict(set)  # {状态: 前驱状态集合}
        states = self.mdp.getStates()

        for s in states:
            if not self.mdp.isTerminal(s):
                # Traverse all possible actions and collect predecessors
                for action in self.mdp.getPossibleActions(s):
                    # Get the transition state (s') and probability (P(s'|s,a))
                    for trans_state, prob in self.mdp.getTransitionStatesAndProbs(s, action):
                        if prob > 0:  # Only keep predecessors with non-zero transition probabilities
                            predecessors[trans_state].add(s)

        # 2. Initialize an empty priority queue.
        p_queue = util.PriorityQueue()

        # 3. For each non-terminal state s
        for s in states:
            if not self.mdp.isTerminal(s):
                actions = self.mdp.getPossibleActions(s)
                max_q_value = 0.0
                if actions:
                    max_q_value = max(self.computeQValueFromValues(s, action) for action in actions)
                # Find the absolute value of the difference between the current value of s
                diff = abs(self.values[s] - max_q_value)
                # Push s into the priority queue with priority -diff (note that this is negative).
                p_queue.update(s, -diff)

        # 4. For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for _ in range(self.iterations):
            # o If the priority queue is empty, then terminate.
            if p_queue.isEmpty():
                continue;

            # o Pop a state s with the largest diff from the priority queue.
            s = p_queue.pop()

            if not self.mdp.isTerminal(s):
                # o Update s's value (if it is not a terminal state) in self.values.
                actions = self.mdp.getPossibleActions(s)
                if actions:
                    best_q_val = max(self.computeQValueFromValues(s, action) for action in actions)
                    self.values[s] = best_q_val

                # o For each predecessor p of s, do:
                for s in predecessors.get(s, []):
                    if not self.mdp.isTerminal(s):
                        # - Find the absolute value of the difference between the current value of p in self.values and the
                        #     highest Q-value across all possible actions from p (this represents what the value should be);
                        #     call this number diff. Do NOT update self.values[p] in this step.
                        local_actions = self.mdp.getPossibleActions(s)
                        local_max_q = 0.0
                        if local_actions:
                            local_max_q = max(self.computeQValueFromValues(s, action) for action in local_actions)
                        local_diff = abs(self.values[s] - local_max_q)
                        # - If diff > theta, push p into the priority queue with priority -diff (note that this is negative),
                        #     as long as it does not already exist in the priority queue with equal or lower priority.
                        #     As before, we use a negative because the priority queue is a min heap, but we want to prioritize
                        #     updating states that have a higher error.
                        if local_diff > self.theta:
                            p_queue.update(s, -local_diff)

