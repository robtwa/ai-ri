# qlearningAgents.py
# ------------------
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
from collections import defaultdict

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

# Question 6 (4 points): Q-Learning
class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        # todo: QLearningAgent.init()

        # Q(s,a)存储：{状态: {动作: Q值}}
        self.q_values = defaultdict(lambda: defaultdict(float))

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # todo: QLearningAgent.getQValue()
        return self.q_values[state].get(action, 0.0)

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        # todo: QLearningAgent.computeValueFromQValues()
        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return 0.0  #  Terminal state → return 0

        # Get max Q(s,a) over all actions
        max_q = max(self.getQValue(state, action) for action in legal_actions)
        return max_q

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # todo: QLearningAgent.computeActionFromQValues()

        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return None  #  Return None if terminal

        # Build a dictionary of Q(s,a) for each action
        q_values = {action: self.getQValue(state, action) for action in legal_actions}

        # Find the max Q-value
        max_q = max(q_values.values())

        # Find all actions with max Q-value
        best_actions = [action for action, q in q_values.items() if q == max_q]

        # Randomly pick an optimal action
        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        # todo: Q7: QLearningAgent.getAction()

        legal_actions = self.getLegalActions(state)
        if not legal_actions:
            return None  # Terminal state
        """
        With probability ε, randomly pick an action to explore.
        With probability 1−ε, pick the current best action to exploit.
        """
        if util.flipCoin(self.epsilon):
            return random.choice(legal_actions)
        else:
            # With 1−ε probability, pick best action
            return self.computeActionFromQValues(state)

    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # todo: QLearningAgent.update()
        # Q-learning formula: Q(s,a) ← (1 - α) * Q(s,a) + α * (r + γ * max_a' Q(s',a'))
        # 1: Estimate the maximum value of the subsequent state V(s') = max_a' Q(s', a')
        # Compute value of next state using current Q-values
        next_value = self.computeValueFromQValues(next_state)

        # 2: Compute the target Q value = immediate reward + discounted subsequent value
        # Target = r + γ * V(s')
        target_q = reward + self.discount * next_value

        # 3: Get the current estimate of Q(s,a)
        current_q = self.getQValue(state, action)

        # 4: Apply Q-learning update formula
        # Q(s,a) ← (1 - α) * Q(s,a) + α * Target
        new_q = (1 - self.alpha) * current_q + self.alpha * target_q

        # 5: Update the new Q value
        self.q_values[state][action] = new_q

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

# Question 10: Approximate Q-Learning
# python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
# python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic
# python autograder.py -q q10
class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # todo: ApproximateQAgent.getQValue()
        # {feature_name: value}
        features = self.featExtractor.getFeatures(state, action)

        # Q(s,a) = ∑ features[f] * weights[f]
        total = 0
        for f in features:
            total += features[f] * self.weights[f]
        return total

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        # todo: ApproximateQAgent.update()
        # weight[f] ← weight[f] + α * (target - Q(s,a)) * f(s,a)
        # weight[f] ← weight[f] + α * δ * f(s,a)

        # Compute the maximum Q value for the next state (greedy strategy)
        max_q_value = self.getValue(nextState)

        # Compute the TD target value: reward + γ × max Q(s', a')
        target = reward + self.discount * max_q_value

        # Current Q-value estimate
        curr_q_value = self.getQValue(state, action)

        # δ: Temporal Difference (TD) error
        td = target - curr_q_value

        # Extract feature vector for (state, action)
        features = self.featExtractor.getFeatures(state, action)
        for f in features:
            # w[f] ← w[f] + α × δ × f(s,a)
            self.weights[f] += self.alpha * td * features[f]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            # todo: ApproximateQAgent.final()
            for f, w in self.weights.items():
                print(f"{f:>40} : {w: .2f}")

