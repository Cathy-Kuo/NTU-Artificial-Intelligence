# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Initial the score from successorGameState.
        score = successorGameState.getScore()

        # Get penalty if the pacman stop.
        if action == Directions.STOP:
            score -= 5

        # Get the distance between agent and ghost.
        ghost_dist = manhattanDistance(newPos, newGhostStates[0].getPosition())

        # I set 10 for the initail distance of food list to prevent the value error of min().
        food_dist = [10]
        food_list = newFood.asList()

        # Get number of food from currentGameState.
        current_food = len(currentGameState.getFood().asList())
        # Create distance list between pacman and food from all the remaining food.
        for food in food_list:
            food_dist.append(manhattanDistance(food,newPos))
          
        # Get score if the pacman get food.
        if current_food < len(food_list):
            score += 50
        # Pacman should move to the position of food.
        # For the last state, the successorGameState.getFood() is empty, so the initial food_dist can prevent the value error.
        score += (30-min(food_dist))
        
        # Pacman can't touch the ghost.
        if ghost_dist < 2:
            score -= 5

        return score
        
        

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, depth):
            legal_act = gameState.getLegalActions(0)
            if depth > self.depth or not legal_act:
                return self.evaluationFunction(gameState)

            act_cost = []
            for act in legal_act:
                successor = gameState.generateSuccessor(0, act)
                act_cost.append((min_value(successor, 1, depth), act))

            return max(act_cost)

        def min_value(gameState, agentIndex, depth):
            legal_act = gameState.getLegalActions(agentIndex)
            if gameState.isLose() or not legal_act:
                return self.evaluationFunction(gameState)

            val = []
            for act in legal_act:
                for successor in [gameState.generateSuccessor(agentIndex, act)]:
                    if agentIndex == gameState.getNumAgents() - 1:
                        val.append(max_value(successor, depth + 1))
                    else:
                        val.append(min_value(successor, agentIndex + 1, depth))

            return min(val)
        
        best_act = max_value(gameState, 1)[1]
        return best_act

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        Inf = float('inf')
        def max_value(gameState, depth, alpha, beta):
            legal_act = gameState.getLegalActions(0)
            if depth > self.depth or not legal_act:
                return self.evaluationFunction(gameState), Directions.STOP

            value = -Inf
            best_act = Directions.STOP
            for act in legal_act:
                successor = gameState.generateSuccessor(0, act)
                cost = min_value(successor, 1, depth, alpha, beta)[0]
                if cost > value:
                    value = cost
                    best_act = act
                if value > beta:
                    return value, best_act
                alpha = max(alpha, value)

            return value, best_act

        def min_value(gameState, agentIndex, depth, alpha, beta):
            legal_act = gameState.getLegalActions(agentIndex)
            if gameState.isLose() or not legal_act:
                return self.evaluationFunction(gameState), Directions.STOP

            value = Inf
            best_act = Directions.STOP
            for act in legal_act:
                successor = gameState.generateSuccessor(agentIndex, act)
                if agentIndex == gameState.getNumAgents() - 1:
                    cost = max_value(successor, depth + 1, alpha, beta)[0]
                else:
                    cost = min_value(successor, agentIndex + 1, depth, alpha, beta)[0]

                if value > cost:
                    value = cost
                    best_act = act
                if alpha > value:
                    return value, best_act
                beta = min(beta, value)

            return value, best_act

        best_act = max_value(gameState, 1, -Inf, Inf)[1]
        return best_act

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, depth):
            legal_act = gameState.getLegalActions(0)
            if depth > self.depth or not legal_act:
                return self.evaluationFunction(gameState), None

            act_cost = []
            for act in legal_act:
                successor = gameState.generateSuccessor(0, act)
                act_cost.append((expected_value(successor, 1, depth)[0], act))

            return max(act_cost)

        def expected_value(gameState, agentIndex, depth):
            legal_act = gameState.getLegalActions(agentIndex)
            if not legal_act or gameState.isLose():
                return self.evaluationFunction(gameState), None

            successors = [gameState.generateSuccessor(agentIndex, act) for act in legal_act]
            act_cost = []
            for successor in successors:
                if agentIndex == gameState.getNumAgents() - 1:
                    act_cost.append(max_value(successor, depth + 1))
                else:
                    act_cost.append(expected_value(successor, agentIndex + 1, depth))
            avg_score = sum(map(lambda x: float(x[0]) / len(act_cost), act_cost))

            return avg_score, None

        best_act = max_value(gameState, 1)[1]
        return best_act

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <Initial the score from current game state. I expect the pacman can move 
      to food, so that I minus the minimum distance between food and the pacman. Moreover, I expect the pacman 
      can move away from the ghost, so that if the distance between them small than 2, the pacman would
      get penalty.>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()

    food_dist = [10]

    current_food = currentGameState.getFood().asList()
    for food in current_food:
        food_dist.append(manhattanDistance(food,pos))
    score += (30-min(food_dist))

    ghost_dist = manhattanDistance(newPos, newGhostStates[0].getPosition())
    if ghost_dist < 2:
        score -= 5

    return score

# Abbreviation
better = betterEvaluationFunction

