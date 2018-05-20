
CONLL_STRUCTURE = ['id', 'orth', 'lemma', 'pos', 'pos2', 'morph', 'parent', 'relation', '_1', '_2', '_', 'as' ]




# class with up and down traverse methods and token requirement checking
class Sentence:

    def parseConllSent(self, conllSent):
        byId = {}
        for line in conllSent:
            # read line:
            for idx, val in enumerate(line.strip().split()):
                if CONLL_STRUCTURE[idx] == 'id':
                    id = int(val)
                    byId[id] = {}
                if CONLL_STRUCTURE[idx] == 'parent':
                    byId[id][CONLL_STRUCTURE[idx]] = int(val)
                else:
                    byId[id][CONLL_STRUCTURE[idx]] = val
        return byId

    def __init__(self, conllSent):
        self.byId = self.parseConllSent( conllSent )
        self.ids = self.byId.keys()

    # here we assume it's the last column
    def getTokenIDsWithAttribute(self, attribute):
        return [Id for Id, tokenDict in self.byId.items() if tokenDict['as']==attribute ]
    

    # check node requirements (as specified in [..] )
    def checkNodeRequirements(self, nodeId, requirementsList):
        if type(nodeId) is list or nodeId==0:
            print "checkNodeRequirements requires int not list:",nodeId
            return None
        for attribute,value in requirementsList.items():
            if not self.byId[nodeId][attribute] == value:
                return False
        return True


    # easy, as each token has only one parent
    def nodesUpFrom(self, nodeId, relation=None):
        # no relation defined, just return parent node
        if relation is None:
            return self.byId[nodeId]['parent']
        # relation defined, check if matches:
        elif self.byId[nodeId]['relation'] == relation:
            return self.byId[nodeId]['parent'] # yes, return parent it
        else:
            return None


    # move up from the nodes we're/might be in, return a collection of possible ids
    def moveUp(self, currentPossibleNodeIds, visitedNodeIds, requiredLabel=None):
        possibleIds = []
        for nodeId in currentPossibleNodeIds:
            nodeup = self.nodesUpFrom(nodeId, requiredLabel)
            if nodeup is not None and nodeup not in visitedNodeIds:
                possibleIds.append( nodeup )
        #return list of possible node ids
        return possibleIds


    def moveDown(self, currentPossibleNodeIds, visitedNodeIds, requiredLabel=None):
        possibleIds = []
        for nodeId in currentPossibleNodeIds:
            # unfortunately have to review all nodes for their parents:
            for childId in self.ids:
                if self.byId[childId]['parent'] == nodeId:
                    if requiredLabel is not None:
                        if self.byId[childId]['relation'] == requiredLabel and childId not in visitedNodeIds:
                            possibleIds.append(childId)
                    else:
                        possibleIds.append(childId)

        #return list of possible node ids
        return possibleIds
