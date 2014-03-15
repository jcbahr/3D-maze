#!/bin/env python2.7

# a test of line intersection for the term project

import copy

################################################################################
##### Point, Line, and Set Helper Functions ####################################
################################################################################

def isNumber(x):
    """Returns True if x is an int, long, or float; False otherwise."""
    return type(x) in (int, long, float)

def sign(x):
    """Returns "+" for non-negative numbers; "-" otherwise."""
    return "+" if (x >= 0) else "-"

def isPoint(p):
    """Returns True if p is of the form (x,y) where x and y are numbers."""
    if (type(p) != tuple):
        return False
    elif (len(p) != 2):
        return False
    return isNumber(p[0]) and isNumber(p[1])

def isSeg(seg):
    """Returns True if seg is of the form (p,q) where p and q are points."""
    if (type(seg) != tuple):
        return False
    elif (len(seg) != 2):
        return False
    return isPoint(seg[0]) and isPoint(seg[1])

def makeSegment(p1,p2):
    return (p1,p2)

def xKey(tuple):
    (x,y) = tuple
    return x

def yKey(tuple):
    (x,y) = tuple
    return y

def getElementFromSet(s):
    """Returns an arbitrary element in a set non-destructively"""
    for val in s:
        return val # breaks

def minMaxXPoint(pointSet):
    """Returns the minimum and maximum points in a set of points with
    respect to the first (x) coordinate."""
    minPoint = min(pointSet, xKey)
    maxPoint = max(pointSet, xKey)
    return (minPoint, maxPoint)

def minMaxYPoint(pointSet):
    """Returns the minimum and maximum points in a set of points with
    respect to the second (y) coordinate."""
    minPoint = min(pointSet, yKey)
    maxPoint = max(pointSet, yKey)
    return (minPoint, maxPoint)

def segmentType(seg):
    if (not isSeg(seg)): return False
    ((x1, y1), (x2, y2)) = seg
    if (x1 == x2):
        return "vert"
    elif (y1 == y2):
        return "horiz"
    else:
        assert(False), "not a rook segment"

################################################################################
##### Line Helper Functions ####################################################
################################################################################

def makeRay(eye, point):
    (eyeX, eyeY) = eye
    (pX, pY) = point
    dx = float(pX - eyeX)
    dy = float(pY - eyeY)
    if (dx == dy == 0):
        assert(False), "cannot make ray from point to self"
    return (eye, (dx, dy))

def makeLine(p1, p2):
    # points are 2-tuples (x,y)
    # lines are (a,b,c) st ax+by=c
    ((x1,y1),(x2,y2)) = (p1,p2)
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    if (dx != 0):
        a = -dy/dx
        b = 1
        c = a*x1 + b*y1
    else: # vertical (slope undefined)
        a = 1
        b = 0
        c = x1
    return (a,b,c)

def isHorizSeg(segment):
    (p1,p2) = segment
    if (not isPoint(p1) or not isPoint(p2)):
        assert(False), "not a segment"
    ((x1,y1),(x2,y2)) = (p1,p2)
    return (y1 == y2)

def isVertSeg(segment):
    (p1,p2) = segment
    if (not isPoint(p1) or not isPoint(p2)):
        assert(False), "not a segment"
    ((x1,y1),(x2,y2)) = (p1,p2)
    return (x1 == x2)

################################################################################
##### Line Intersection Functions ##############################################
################################################################################

# NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE: k <= 1 is NOT SUFFICIENT
# If one point has a k < 1 but the other has a k > 1, DIFFERENT THINGS HAPPEN
# NOTE NOTE NOTE NOTE NOTE NOTE NOTE NOTE

def intersectRayAndVertSegment(ray, segment):
    ((x1,y1), (x2,y2)) = segment
    (eye, (dx, dy)) = ray
    (eyeX, eyeY) = eye
    xSeg = float(x1) # = x2
    if (dx != 0):
        k = (xSeg - eyeX) / dx # vector form: pointOnLine = k*(dx,dy) + eye
        print "k = ", k
        if (k < 0): # wall behind eye
            return None
        elif (k <= 1): # wall behind segment
            yIntercept = k*dy + eyeY
            return ["behind", (xSeg, yIntercept)]
        else:
            yIntercept = k*dy + eyeY
            return (xSeg, yIntercept)
    else: # vertical
        if (xSeg != eyeX):
            # no intersection
            # but we consider it to intersect at pos. or neg. infinity
            return sign(dy) + "infinity"
        else: # collinear
            return segment

def intersectRayAndHorizSegment(ray, segment):
    ((x1,y1), (x2,y2)) = segment
    (eye, (dx, dy)) = ray
    (eyeX, eyeY) = eye
    ySeg = float(y1) # = y2
    if (dy != 0):
        k = (ySeg - eyeY) / dy # vector form: pointOnLine = k*(dx,dy) + eye
        print "k = ", k
        if (k < 0): # wall behind eye
            return None
        elif (k <= 1): # wall behind segment
            xIntercept = k*dx + eyeX
            return ["behind", (xIntercept, ySeg)]
        else:
            xIntercept = k*dx + eyeX
            return (xIntercept, ySeg)
    else: # horizontal
        if (ySeg != eyeY):
            # no intersection
            # but we consider it to intersect at pos. or neg. infinity
            return sign(dx) + "infinity"
        else: # collinear
            return segment

def intersectRayAndRookSegment(ray, segment):
    # does not check if intersection actually lies on the line
    # returns a point of intersection
    #         a segment of intersection
    #      or "+infinity", "-infinity" for no intersection

    ((x1,y1), (x2,y2)) = segment
    if ((x1 != x2) and (y1 != y2)):
        assert(False), "not a rook segment"
    if (x1 == x2):
        return intersectRayAndVertSegment(ray, segment)
    else:
        return intersectRayAndHorizSegment(ray, segment)


def intersectWalls(wall, seg):
    # given two rook walls, finds the predicted intersection
    # this is only for when one point is "behind" and the other is not
    # note that wall and seg are both rook segments
    #  their different names are for clarity only

    ((wallX1, wallY1), (wallX2, wallY2)) = wall
    ((segX1, segY1), (segX2, segY2)) = seg
    wallType = segmentType(wall)
    segType = segmentType(seg)
    if (segType == wallType):
        assert(False), "wall and seg not perpendicular"
    elif (wallType == "horiz"):
        return (segX1, wallY1)
    else:
        return (wallX1, segY1)





################################################################################
##### Line Obstruction Functions ###############################################
################################################################################

def obstructViaIntersections(intersection, seg):
    obstructions = set()
    ((segX1, segY1), (segX2, segY2)) = seg
    segType = segmentType(seg)
    maxY = max(segY1, segY2)
    minY = min(segY1, segY2)
    maxX = max(segX1, segX2)
    minX = min(segX1, segX2)
    if (isSeg(intersection)):
        obstructions = set([intersection])
    elif (isPoint(intersection)):
        obstructions.add(intersection)
    elif (type(intersection) == str):
        if (segType == "vert"):
            if (intersection[0] == "+"):
                obstructions.add((segX1, maxY))
            else:
                obstructions.add((segX1, minY))
        else: # segType == "horiz"
            if (intersection[0] == "+"):
                obstructions.add((maxX, segY1))
            else:
                obstructions.add((minX, segY1))
    elif (type(intersection) == list): # ["behind", intersection]
        # calculate nearest seg edge
        return ["behind", intersection]
        # FIXME
        # WHAT DO I DO HERE
        # I should return a list with a "behind" keyword and
        #  the closest point of the segment to the intersection
        # NOTE: SHOULD IT BE CLOSEST?
        # that way, if I get two behinds, I can give no obstructions
        # otherwise, make the closest the obstruction
        # NOTE: It should be the intersection!
    return obstructions


def obstructSegViaSeg(eye, wall, seg):
    # "wall" is just shorthand for an obstructing segment
    # when one point is behind the wall and the other is in front,
    #  we instead obstruct with the front one and the point of intersection

    obstructions = set()
    (wallP1, wallP2) = wall
    if ((eye == wallP1) or (eye == wallP2)):
        assert(False), "collision with wall edge"
    ray1 = makeRay(eye, wallP1)
    intersection1 = intersectRayAndRookSegment(ray1, seg)
    ray2 = makeRay(eye, wallP2)
    intersection2 = intersectRayAndRookSegment(ray2, seg)
    print "rays:", ray1, ray2
    obstruct1 = obstructViaIntersections(intersection1, seg)
    obstruct2 = obstructViaIntersections(intersection2, seg)
    if (type(obstruct1) == type(obstruct2) == list): # both behind
        obstruct1 = obstruct2 = set()
    elif ((type(obstruct1) == list) and (type(obstruct2) != list)):
        # 1st is behind
        obstruct1 = set([intersectWalls(wall, seg)])
    elif ((type(obstruct1) != list) and (type(obstruct2) == list)):
        # 2nd is behind
        obstruct2 = set([intersectWalls(wall, seg)])
    print "obs:", obstruct1, obstruct2
    obstructions = obstructions.union(obstruct1, obstruct2)
    return [seg, obstructions]

def chopHorizSegWithObstructions(segWithObstructions):
    (segment, obstructions) = segWithObstructions
    if (len(obstructions) == 0):
        return set([segment])
    ((x1,y1),(x2,y2)) = segment
    (minX, maxX) = (min(x1,x2), max(x1,x2))
    (minY, maxY) = (min(y1,y2), max(y1,y2))
    (minPoint,maxPoint) = minMaxXPoint(obstructions)
    if ((minPoint[0] <= minX) and (maxPoint[0] >= maxX)):
        # segment completely obscured
        print "completely"
        return set()
    elif ((minPoint[0] <= minX) and (maxPoint[0] >= minX)):
        # min not visible, but max is
        # (y1 == y2) since horizontal segment
        print "min not visible, max is"
        return set([(maxPoint, (maxX, y1))])
    elif ((minPoint[0] <= maxX) and (maxPoint[0] >= maxX)):
        # max not visible, but min is
        # (y1 == y2) since horizontal segment
        print minPoint, minX, maxPoint, maxX
        print "max not visible, min is"
        return set([(minPoint, (minX, y1))])
    elif ((minPoint[0] > minX) and (maxPoint[0] < maxX)):
        # segment centrally obscured
        # returning a set of the two segments
        seg1 = ((minX, y1), minPoint)
        seg2 = (maxPoint, (maxX, y1))
        print "centrally obscured"
        return set([seg1, seg2])
    else:
        print obstructions
        print minPoint[0], minX
        print maxPoint[0], maxX
        print "else"
        return set([segment])

def chopVertSegWithObstructions(segWithObstructions):
    (segment, obstructions) = segWithObstructions
    if (len(obstructions) == 0):
        return set([segment])
    ((x1,y1),(x2,y2)) = segment
    (minX, maxX) = (min(x1,x2), max(x1,x2))
    (minY, maxY) = (min(y1,y2), max(y1,y2))
    (minPoint,maxPoint) = minMaxYPoint(obstructions)
    if ((minPoint[1] <= minY) and (maxPoint[1] >= maxY)):
        # segment completely obscured
        return set()
    elif ((minPoint[1] <= minY) and (maxPoint[1] >= minY)):
        # min not visible, but max is
        # (x1 == x2) since vertical segment
        return set([(maxPoint, (x1, maxY))])
    elif ((minPoint[1] <= maxY) and (maxPoint[1] >= maxX)):
        # max not visible, but min is
        # (x1 == x2) since vertical segment
        return set([(minPoint, (x1, minY))])
    elif ((minPoint[1] > minY) and (maxPoint[1] < maxY)):
        # segment centrally obscured
        # returning a set of the two segments
        seg1 = ((x1, minY), minPoint)
        seg2 = (maxPoint, (x1, maxY))
        return set([seg1, seg2])
    else:
        return set([segment])

def chopSegWithObstructions(segWithObstructions):
    # we assume that the segment is a rook segment
    # we also assume that obstructions are on the line
    (segment, obstructions) = segWithObstructions
    if (segment in obstructions):
        # something obscured entire segment
        return set()
    elif (isHorizSeg(segment)):
        return chopHorizSegWithObstructions(segWithObstructions)
    elif (isVertSeg(segment)):
        return chopVertSegWithObstructions(segWithObstructions)
    else:
        assert(False), "not a rook segment"


################################################################################
##### Visibility of Segment ####################################################
################################################################################

def obstructSegViaSetOfSegs(eyePoint, setOfSegs, segment):
    print "obstructSegViaSetOfSegs"
    print "on: ", segment, "\n\n"
    obSegSet = set([segment])
    newObSegSet = set()
    # this is a sort of unordered queue using sets
    for blockingSeg in setOfSegs:
        print "\nchecking the blocker: ", blockingSeg
        for segPiece in obSegSet:
            print "against:",segPiece
            segWithObstructions = obstructSegViaSeg(eyePoint,
                                                    blockingSeg,
                                                    segPiece)
            # set() is the set of obstructions, but it's empty since we chop
            #  before we obscure again
            chopped = chopSegWithObstructions(segWithObstructions)
            print "chopped:", chopped
            newObSegSet = newObSegSet.union(chopped)
        obSegSet = newObSegSet
        print obSegSet
        newObSegSet = set()
    print "-------------\n\n"
    return obSegSet

def obstructSegs(eyePoint, setOfSegs):
    # more optimization can be done by only obstructing farther segs
    #  by closer segs, but this is algorithmically difficult for the moment
    # this is currently O(n**2), but that optimization could cut in half the
    #  number of obstructing checks, but it would remain O(n**2)
    visible = set()
    for seg in setOfSegs:
        # NOTE: Find improvement over copy.copy
        remainingSegs = copy.copy(setOfSegs)
        remainingSegs.remove(seg)
        visible = visible.union(obstructSegViaSetOfSegs(eye,
                                                        remainingSegs,
                                                        seg))
    return visible
                                                    

                                                                
        




################################################################################
##### TIMING ###################################################################
################################################################################

import time

def timeFnCall(f, n, m):
    # call f(n) and return time in ms
    # Actually, since one call may require less than 1 ms,
    # we'll keep calling until we get at least 1 secs,
    # then divide by # of calls we had to make
    calls = 0
    start = end = time.time()
    while (end - start < 1):
        f(n, m)
        calls += 1
        end = time.time()
    return float(end - start)/calls*1000 #(convert to ms)




################################################################################
##### TESTING ##################################################################
################################################################################



        



eye = (5.2,1.2)
segs = set([((1,1),(6,1)),
            ((1,1),(1,6)),
            ((1,6),(6,6)),
            ((6,1),(6,6)),
            ((5,1),(5,2)),
            ((5,2),(3,2)),
            ((2,2),(2,4)),
            ((2,3),(5,3)),
            ((1,4),(2,4)),
            ((5,4),(6,4)),
            ((3,4),(3,5)),
            ((5,4),(5,5)),
            ((4,4),(4,6)),
            ((2,5),(4,5)),
            ((2,5),(2,6))])

#eye = (1,1)
#segs = set([((2,2),(2,4)),
#            ((2,3),(3,3))])

#print obstructSegViaSetOfSegs(eye, s, obSeg)


#print chopSegWithObstructions(obstructSegViaSeg(eye, blockingSeg, seg))
#print obstructSegWithObstructionsViaSeg(eye, blockingSeg, [seg,set()])
#print makeLine(eye, (1,1))

from Tkinter import *
root = Tk()
canvas = Canvas(root, width=500, height=500)
canvas.pack()

#canvas.create_text(50*eye[0], 50*eye[1], text=".", fill="red", font="Times 18")
#canvas.create_text(50*eye[0], 50*eye[1], text=".", fill="red", font="Times 10")
#canvas.create_text(50*eye[0], 50*eye[1], text=".", fill="red", font="Times 2")
canvas.create_line(50*eye[0]-1, 50*eye[1]-1, 50*eye[0]+1, 50*eye[1]+1, fill="red", width=2)

#canvas.create_line(50*eye[0], 50*eye[1], 50*seg1[0][0], 50*seg1[0][1], fill="red", width=2)
#canvas.create_line(50*eye[0], 50*eye[1], 50*seg1[1][0], 50*seg1[1][1], fill="red", width=2)
#canvas.create_line(50*eye[0], 50*eye[1], 50*seg2[0][0], 50*seg2[0][1], fill="red", width=2)
#canvas.create_line(50*eye[0], 50*eye[1], 50*seg2[1][0], 50*seg2[1][1], fill="red", width=2)
#canvas.create_line(50*eye[0], 50*eye[1], 50*seg3[0][0], 50*seg3[0][1], fill="red", width=2)
#canvas.create_line(50*eye[0], 50*eye[1], 50*seg3[1][0], 50*seg3[1][1], fill="red", width=2)
for seg in segs:
    canvas.create_line(50*seg[0][0], 50*seg[0][1], 50*seg[1][0], 50*seg[1][1])

colors = ["blue", "green", "red", "yellow", "cyan", "orange", "magenta", "black", "purple", "brown", "white"]
i = 0
visible = obstructSegs(eye, segs)
print "visible = ",visible
for s in visible:
    canvas.create_line(50*s[0][0], 50*s[0][1], 50*s[1][0], 50*s[1][1], fill=colors[i], width=3)
    i += 1
    i %= 11

#for s in obstructSegViaSetOfSegs(eye, set([seg2]), seg1):
#    canvas.create_line(50*s[0][0], 50*s[0][1], 50*s[1][0], 50*s[1][1], fill=colors[i], width=3)
#    i += 1

root.mainloop()




