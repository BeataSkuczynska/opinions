__author__ = 'aleks'

import pyparsing
from pyparsing import Word, Literal, alphanums, ZeroOrMore, Suppress, Optional, Group, oneOf
import pickle
from sentence import *

class SentencePatternMatcher():

    equals = Literal('=')
    word = Word(alphanums)
    tagAndVal = Group(word.setResultsName("tag") + Suppress(":") +word.setResultsName("val")).setResultsName("tagAndVal")  + Suppress(Optional(";"))
    token = Suppress("[") + Group(ZeroOrMore(tagAndVal)).setResultsName("tagAndVals")  +  Suppress("]")

    edge = oneOf(["<", ">"])+word.setResultsName("label")

    # parse [tag1:val1;tag2:val2] and return python dictionary
    def parseNode(self, tokenString):
        try:
            result = self.token.parseString(tokenString)
        except pyparsing.ParseException:
            print "Path parsing exception: Malformed token string: ",tokenString
        requirementsDict = {}
        #print "result:",result
        for r in result.tagAndVals:
            #print "r:",r
            #print "tagAndVal.tag=",r.tag
            #print "tagAndVal.val=",r.val
            requirementsDict[r.tag] = r.val
        return requirementsDict
        #print "tagAndVal/s:",result.tagAndVals.tagAndVal

    # return label from edge definition
    def parseTag(self, tagString):
        try:
            result = self.edge.parseString(tagString)
        except pyparsing.ParseException:
            print "Path parsing exception: Malformed edge string: ",tagString
        return result.label


    # check sentence against pattern
    def matchPathInSentence(self, startNodeIds, sgpath, s):

        tokens = sgpath.split(' ')
        visitedNodeIds = currentPossibleNodeIds = startNodeIds
        match = 1

        for token in tokens:

            # we are now at token:
            if token.startswith("["):
                requirementsList = self.parseNode(token)
                # check all possible current nodes for complying with requirements:
                validIds = []
                for nodeId in currentPossibleNodeIds:
                    if s.checkNodeRequirements(nodeId, requirementsList):
                        validIds.append(nodeId)
                # no node meets those requirements:
                if len(validIds) == 0:
                    match=0 ; break
                else:
                    currentPossibleNodeIds = validIds
                    visitedNodeIds.extend( currentPossibleNodeIds )

            # we are now at edge indicating upward
            elif token.startswith("<"):
                requiredLabel = self.parseTag(token)
                # get nodes with this label:
                currentNodeIds = s.moveUp(currentPossibleNodeIds, visitedNodeIds, requiredLabel)
                if currentNodeIds is None:  # label did not match
                    match=0 ; break
                else:
                    currentPossibleNodeIds = currentNodeIds

            # we are now at edge indicating downward
            elif token.startswith(">"):
                requiredLabel = self.parseTag(token)
                currentNodeIds = s.moveDown(currentPossibleNodeIds, visitedNodeIds, requiredLabel)
                if currentNodeIds is None:  # label did not match
                    match=0 ; break
                else:
                    currentPossibleNodeIds = currentNodeIds

        if match==0:
            currentPossibleNodeIds = None

        return currentPossibleNodeIds

