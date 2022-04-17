#!/usr/bin/python

from audioop import avg
from calendar import c
from collections import defaultdict
from email.policy import default
import random
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    result = dict()
    
    for word in x.split():
        if(not result.get(word)):
            result[word] = 0
        result[word] += 1

    return result
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)

    def predictor(x):
        if(dotProduct(weights, featureExtractor(x)) >= 0):
            return 1
        return -1

    for _ in range(numEpochs):
        for x, y in trainExamples:
            features = featureExtractor(x)
            
            if(dotProduct(weights, features)*y >= 1):
                continue

            # loss = features*-y
            increment(weights, -eta*-y, features)
        
        # print('Traning loss', evaluatePredictor(trainExamples, predictor))
        # print('validation loss', evaluatePredictor(validationExamples, predictor))

    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = {}
        for key in weights:
            phi[key] = random.random()
        
        y = 1
        if(dotProduct(weights, phi) < 0):
            y = 0

        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x = x.replace(' ', '')
        result = dict()
        for i in range(len(x)-n+1):
            substr = x[i:i+n]
            if(substr not in result):
                result[substr] = 0
            
            result[substr] += 1

        return result
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 5: k-means
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)
    
    centers: List[float] = []
    assignments: List[int] = [None for _ in examples]
    totalCost = 0

    distanceCache = dict()
    def distance(i, j):
        f1 = examples[i]
        f2 = centers[j]

        f1Key = 'example-{}'.format(i)
        f2Key = 'center-{}'.format(j)

        if(f1Key not in distanceCache):
            distanceCache[f1Key] = dotProduct(f1, f1)
        
        # if(f2Key not in distanceCache):
        distanceCache[f2Key] = dotProduct(f2, f2)

        return distanceCache[f1Key] + distanceCache[f2Key] - 2*dotProduct(f1, f2)

    def initCenters():
        centers.clear()
        while(len(centers) < K):
            i = random.randint(0, len(examples)-1)
            if(examples[i] not in centers):
                centers.append(examples[i].copy())
    
    def assignCenter(example_i):
        minCenter = 0
        for i in range(1, len(centers)):
            if(distance(example_i, i) < distance(example_i, minCenter)):
                minCenter = i
        return minCenter

    def updateCenter(center_i, example_is):
        center = centers[center_i]

        count = defaultdict(int)
        avgDict = defaultdict(float)

        for example_i in example_is:
            example = examples[example_i]
            for key in example:
                count[key] += 1
                avgDict[key] += example[key]

        for key in count:
            avgDict[key] /= count[key]

        update = False
        for key in avgDict:
            if(key not in center or avgDict[key] != center[key]):
                update = True
                center[key] = avgDict[key]
        
        if(update):
            distanceCache[f'center-{center_i}'] = dotProduct(center, center)

    def updateCenters():
        totalCost = 0
        centerDict = defaultdict(list)

        for i in range(len(assignments)):
            center_i = assignments[i]
            example = examples[i]

            totalCost += distance(i, center_i)
            centerDict[center_i].append(i)

        for center_i in centerDict:
            center = centers[center_i]
            updateCenter(center_i, centerDict[center_i])
        
        return update
            

    initCenters()
    for _ in range(maxEpochs):
        totalCost = 0
        update = False
        for i in range(0, len(examples)):
            example = examples[i]
            newCenter_i = assignCenter(i)
            if(assignments[i] != newCenter_i):
                update = True
            assignments[i] = newCenter_i

            totalCost += distance(i, newCenter_i)

        if not update:
            break

        updateCenters()
    
    return centers, assignments, totalCost

    # END_YOUR_CODE
