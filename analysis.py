# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question0():
    # This should be an integer greater than zero
    hoursWorked = 30
    return hoursWorked

# V(s) = max_a ∑_s' P(s' | s, a) × [R(s, a, s') + γ × V(s')]
# Q(s,a) = ∑_s' P(s'|s,a) × [R(s,a,s') + γ × max_a' Q(s', a')]
# γ or discount:
#   - γ × V(s')
#   - The larger the value, the less penalty is imposed, and the agent is more willing to take the longer route
#     (0.9 * 1 = 0.9)
#   - The smaller the value, the greater the penalty, and the agent is less willing to take the longer route
#     (0.1 * 1 = 0.1)
# P(s'|s,a) or Noise:
#   - Action Error
#   - The probability that, in state s, the agent executes action a and then transitions to state s' .
#   - The larger the value, the more likely the agent is to make an error when executing the action
#      -> Afraid of falling off the cliff edge
# R(s, a, s') or living reward :
#   - Reward for each step taken
#   - The smaller the value, the more eager the agent is to reach the destination quickly ( -1 * 10 = -10 )
#   - The larger the value, the more the agent desires to take more steps without reaching the end ( 10 * 10 = 100 )

def question2():
    """
    Question 2 (1 point): Bridge Crossing Analysis
    With the default discount of 0.9 and the default noise of 0.2, the optimal policy does not cross the bridge.
    Change only ONE of the discount and noise parameters so that the optimal policy causes the agent to attempt to cross
    the bridge.
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    return answerDiscount, answerNoise


def question3a():
    """
    Question 3 (5 points): Policies
    a. Prefer the close exit (+1), risking the cliff (-10)

    Prefer the close exit (+1) -> Finish quickly, take fewer steps
    risking the cliff (-10)    -> Have smaller movement error, dare to walk on the cliff
    """
    answerDiscount = 0.3
    answerNoise = 0.0
    answerLivingReward = -0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    """
    Question 3 (5 points): Policies
    b. Prefer the close exit (+1), but avoiding the cliff (-10)

    Preferring the close exit (+1)  -> Finish quickly, take fewer steps
    But avoiding the cliff (-10)    -> The movement error should be slightly larger, stay away from the cliff
    """
    answerDiscount = 0.3
    answerNoise = 0.3
    answerLivingReward = -0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    """
    Question 3 (5 points): Policies
    c. Prefer the distant exit (+10), risking the cliff (-10)

    Prefer the distant exit (+10)   -> Getting farther away from the exit earns more points
    risking the cliff (-10)         -> Have smaller movement error, dare to walk on the cliff
    """
    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = -0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    """
    Question 3 (5 points): Policies
    d. Prefer the distant exit (+10), avoiding the cliff (-10)

    Prefer the distant exit (+10)   -> Getting farther away from the exit earns more points
    avoiding the cliff (-10)        -> The movement error should be slightly larger, stay away from the cliff
    """
    answerDiscount = 0.9
    answerNoise = 0.3
    answerLivingReward = -0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    """
    Question 3 (5 points): Policies
    e. Avoid both exits and the cliff (so an episode should never terminate)

    Avoid both exits            -> The more exits to avoid, the more points to get.
    avoiding the cliff (-10)    -> The movement error should be slightly larger, stay away from the cliff
    """
    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = 1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    # answerEpsilon = 0.05
    # answerLearningRate = 0.5
    # return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
