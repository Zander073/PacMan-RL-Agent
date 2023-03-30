# myTeam.py
# ---------
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

from typing import Literal
from copy import deepcopy
from os import path
import json
from captureAgents import CaptureAgent
import random
import time
import util
from game import Directions
import game
from util import Counter, nearestPoint
from capture import GameState

#################
# Team creation #
#################


class Constants:
    TRAINING = False
    WEIGHTS_FILE = "weights_pacmen.json"
    WEIGHTS = {"offense": {}, "defense": {}}
    EPSILON = 0.05
    ALPHA = 0.1
    DISCOUNT = 0.1
    FEATURES = {"offense": ["food_eaten", "food_dist", "closer_to_ghost", "dist_home", "successor_score", "stop"],
                "defense": ["on_defense", "dist_to_invader", "num_invaders", "ate_invader", "dist_to_center", "stop", "reverse"]}


def createTeam(firstIndex, secondIndex, isRed, first='OffensiveAgent', second='DefensiveAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class QLearningAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.initial_state = gameState.getAgentPosition(self.index)
        self.weights = self.initializeWeights()
        self.prevAction = Directions.STOP
        self.numFoodEaten = 0
        self.numFood = len(self.getFood(gameState).asList())
        self.explore = True
        self.foodToReturn = 1
        if gameState.isOnRedTeam(self.index):
            self.offset = -1
        else:
            self.offset = 1
        self.center = ((gameState.data.layout.width / 2) + self.offset, (gameState.data.layout.height / 2) + self.offset)

    def computeActionFromQValues(self, state: GameState):
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        legalActions = state.getLegalActions(self.index)
        if len(legalActions) == 0:
            return None
        best_action = [None, float("-inf")]
        for action in legalActions:
            qVal = self.getQValue(state, action)
            if qVal == best_action[1]:
                best_action[0] = random.choice([action, best_action[0]])
                best_action[1] = qVal
            elif qVal > best_action[1]:
                best_action = [action, qVal]
        return best_action

    def chooseAction(self, state: GameState):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.
        """
        legalActions = state.getLegalActions(self.index)
        currentState: GameState = self.getCurrentObservation()
        # action = self.prevAction
        prev: GameState = self.getPreviousObservation()
        action = self.computeActionFromQValues(state)[0]
        if prev:
            reward = self.getReward(prev, self.prevAction, currentState)
            if Constants.TRAINING:
                self.update(prev, action, currentState, reward)
        if len(legalActions) == 0:
            return None
        epsGreedy = Constants.EPSILON
        if Constants.TRAINING == False:
            epsGreedy = 0.0
        if util.flipCoin(epsGreedy):
            action = random.choice(legalActions)
        if state.getAgentState(self.index).numCarrying >= self.foodToReturn and isinstance(self, OffensiveAgent):
            action = self.goHome(state)
            self.prevAction = action
            temp = self.getSuccessor(state, action)
            if temp.getScore() > state.getScore():
                self.numFoodEaten = 0
            return action
        self.prevAction = action
        return action

    def getQValue(self, state: GameState, action: Literal) -> (float):
        """
        Should return Q(state,action) = <w> * <featureVector>,
        where * is the dotProduct operator
        """
        featureVector: Counter = self.getFeatures(state, action)
        dotProd: float = featureVector * self.weights
        return dotProd

    def update(self, state: GameState, action: Literal, nextState: GameState, reward):
        features = self.getFeatures(state, self.prevAction)
        for f in features.keys():
            if reward[f] == 0:
                continue
            sample = reward[f] + (Constants.DISCOUNT * self.getQValue(nextState, action))
            diff = sample - self.getQValue(state, self.prevAction)
            self.weights[f] += (Constants.ALPHA * diff)
        self.saveWeights()

    def initializeWeights(self):
        onOffense = isinstance(self, OffensiveAgent)
        weights = util.Counter()
        # Weights saved so load them in
        if path.exists('offensive_weights.json') and path.exists('defensive_weights.json'):
            if onOffense:
                # If on offense, load offensive weights
                file = open("offensive_weights.json", "r")
                weightsDict = json.load(file)["offense"]
                file.close()
            else:
                # If on defense, load defensive weights
                file = open("defensive_weights.json", "r")
                weightsDict = json.load(file)["defense"]
                file.close()
            for key, value in weightsDict.items():
                weights[key] = value
            return weights
        # No weights saved so initialize them:
        else:
            o = open("offensive_weights.json", 'w')
            json.dump({"offense": {"food_eaten": 0.01, "closer_to_ghost": 0.01, "closer_to_food": 0.01, "stopped": .01}, "defense": {}}, o)
            d = open("defensive_weights.json", 'w')
            json.dump({"offense": {}, "defense": {"on_defense": 0.01, "closer_to_invader": 0.01, "stopped": 0.01}}, d)

        return weights

    # Saves weights to JSON file for offense and defense respectively.
    def saveWeights(self):
        onOffense = isinstance(self, OffensiveAgent)
        weights = deepcopy(Constants.WEIGHTS)
        if onOffense:
            file = open("offensive_weights.json", "w")
            weights["offense"] = self.weights
            json.dump(weights, file)
            file.close()
        else:
            file = open("defensive_weights.json", "w")
            weights["defense"] = self.weights
            json.dump(weights, file)
            file.close()

    def getWeights(self) -> (Counter):
        return self.weights

    def getSuccessor(self, gameState: GameState, action: Literal) -> (GameState):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def goHome(self, gameState):
        actions = gameState.getLegalActions(self.index)
        bestDist = 9999
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            dist = self.getMazeDistance(self.center, pos2)
            if dist < bestDist:
                bestAction = action
                bestDist = dist
        return bestAction

    # Returns the minimum distance to an enemy and which enemy
    def getDistanceToEnemy(self, gameState):
        pos = gameState.getAgentState(self.index).getPosition()
        opponents = self.getOpponents(gameState)
        dist = 9999
        for opponent in opponents:
            if gameState.getAgentState(self.index).isPacman and not gameState.getAgentState(opponent).isPacman:
                return -1
            if not (gameState.getAgentPosition(opponent) == None):
                enemyPos = gameState.getAgentPosition(opponent)
                temp_dist = self.distancer.getDistance(pos, enemyPos)
                if temp_dist < dist:
                    dist = temp_dist
        return dist


class OffensiveAgent(QLearningAgent):
    # "food_eaten", "food_dist", "closer_to_ghost", "dist_home", "successor_score", "stop"
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # The agent's currnet position in the current game's state:
        currentPos = gameState.getAgentState(self.index).getPosition()

        # The agent's possible position in a game state' given an action:
        successorPos = successor.getAgentPosition(self.index)

        # Number of food in present and successor states:
        currentFood = self.getFood(gameState).asList()
        successorFood = self.getFood(successor).asList()

        # Check if food was eaten
        if len(currentFood) > len(successorFood):
            features["food_eaten"] = 1
        else:
            features["food_eaten"] = 0

        # Check if moved closer to food:
        foodList = self.getFood(successor).asList()
        minDistance = min([self.getMazeDistance(currentPos, food) for food in foodList])
        minDistanceSuccessPos = min([self.getMazeDistance(successorPos, food) for food in foodList])
        if minDistanceSuccessPos < minDistance:
            features['closer_to_food'] = 1

        # Check if closer to ghost:
        # d1 = self.getDistanceToEnemy(gameState)
        # d2 = self.getDistanceToEnemy(successor)

        # if d2 == -1:
        #     print("hello")


        # if d2 <= 4 and d2 == -1:
        #     features["closer_to_ghost"] = 1
        # else:
        #     features["closer_to_ghost"] = 0

        # Check for stopping:
        if action == Directions.STOP:
            features["stopped"] = 1
        else:
            features["stopped"] = 0

        return features

    def getReward(self, prevState, action, gameState):
        reward = {"food_eaten": 0, "closer_to_food": 0, "stopped": 0, "living": 0, "closer_to_ghost": 0}
        features = self.getFeatures(prevState, action)

        if features["food_eaten"] == 1:
            self.numFoodEaten += 1
            reward["food_eaten"] += 10

        if features["closer_to_food"] == 1:
            reward["closer_to_food"] += 1

        if features["closer_to_ghost"] == 1:
            reward["closer_to_ghost"] -= 5

        if features["stopped"] == 1:
            reward["stopped"] -= 100

        # living reward
        # for key in reward.keys():
        #     reward[key] -= .05

        return reward


class DefensiveAgent(QLearningAgent):
    # "on_defense", "closer_to_invader", "ate_invader", "closer_to_center", "stopped"
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # The agent's currnet position in the current game's state:
        currentPos = gameState.getAgentState(self.index).getPosition()

        # The agent's possible position in a game state' given an action:
        successorPos = successor.getAgentPosition(self.index)

        # On defense:
        features['on_defense'] = 1
        if successor.getAgentState(self.index).isPacman:
            features['on_defense'] = 0

        # Closer to invader:
        succEnemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        currEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        currInvaders = [a for a in currEnemies if a.isPacman and a.getPosition() != None]
        succInvaders = [a for a in succEnemies if a.isPacman and a.getPosition() != None]
        if len(currInvaders) and len(succInvaders) > 0:
            if (min([self.getMazeDistance(currentPos, a.getPosition()) for a in currInvaders]) > min([self.getMazeDistance(successorPos, a.getPosition()) for a in succInvaders])):
                features['closer_to_invader'] = 1
            else:
                features['closer_to_invader'] = 0
        else:
            features['closer_to_invader'] = 0

        # if len(succEnemies) < len(self.getOpponents(gameState)):
        #     features["ate_invader"] = 1
        # else:
        #     features["ate_invader"] = 0

        # Closer to center:
        # d1 = self.getMazeDistance(currentPos, self.center)
        # d2 = self.getMazeDistance(successorPos, self.center)
        # if d2 < d1:
        #     features["closer_to_center"] = 1
        # else:
        #     features["closer_to_center"] = 0
        # features["closer_to_center"] = self.getMazeDistance(currentPos, self.center)

        # Check for stopping:
        if action == Directions.STOP:
            features["stopped"] = 1
        else:
            features["stopped"] = 0
        return features

    def getReward(self, prevState, action, gameState):
        reward = {"on_defense": 0, "closer_to_invader": 0, "ate_invader": 0, "closer_to_center": 0, "stopped": 0, "living": 0}

        features = self.getFeatures(prevState, action)

        if features["on_defense"] == 1:
            reward["on_defense"] += 10

        if features["closer_to_invader"] == 1:
            reward["closer_to_invader"] += 30

        # if features["ate_invader"] == 1:
        #     reward["ate_invader"] += 100

        # if features["closer_to_center"] == 1:
        #     reward["closer_to_center"] += (1-features["closer_to_center"])*10

        if features["stopped"] == 1:
            reward["stopped"] -= 50

        # # living reward
        # for key in reward.keys():
        #     reward[key] -= 1

        return reward
