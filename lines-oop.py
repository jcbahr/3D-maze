#!/bin/env pypy

# lines-oop.py
# line intersection (an object oriented version)

import copy
import math
from Tkinter import *
import random
import time
#random.seed(41)

CYCLE_AMOUNT = 5 # higher number -> fewer cycles

################################################################################
##### Point & Seg Helper Functions #############################################
################################################################################

def flipCoin():
    # taken from:
    # kosbie.net/cmu/fall-12/15-112/handouts/notes-recursion/mazeSolver.py
    return random.choice([True, False])

def smallChance():
    choices = [True]
    choices.extend([False]*CYCLE_AMOUNT)
    print choices
    return random.choice(choices)


def withinEp(x, y):
    """Returns True if x and y are within some predefined epsilon: 0.001"""
    epsilon = 0.0001
    return abs(x-y) < epsilon

def isNumber(x):
    """Returns True if x in an int, long, or float; False otherwise"""
    return type(x) in (int, long, float)

def sign(x):
    """Returnes "+" for non-negative numbers; "-" otherwise"""

def xKey(point):
    return point.x

def yKey(point):
    return point.y

def getElementFromSet(s):
    """Non-destructively returns an arbitrary element from a set"""
    for val in s:
        return val # breaks

def extremeX(pointSet):
    minPoint = min(pointSet, key=xKey)
    maxPoint = max(pointSet, key=xKey)
    return (minPoint, maxPoint)

def extremeY(pointSet):
    minPoint = min(pointSet, key=yKey)
    maxPoint = max(pointSet, key=yKey)
    return (minPoint, maxPoint)

################################################################################
##### Point, Seg, Ray Classes ##################################################
################################################################################

class Point(object):
    def __init__(self, x, y):
        if (not isNumber(x) or not isNumber(y)):
            assert(False), "cannot make a point from non-numbers"
        self.x = x
        self.y = y

    def __eq__(self, other):
        return (withinEp(self.x, other.x) and withinEp(self.y, other.y))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "(%f,%f)" % (self.x, self.y)

    def __repr__(self):
        return "Point(%f,%f)" % (self.x, self.y)

    def __hash__(self):
        hashables = (self.x, self.y)
        return hash(hashables)

class Seg(object):
    def __init__(self, p1, p2):
        # sanity check
        if not (type(p1) == type(p2) == Point):
            assert(False), "cannot make a seg from nonpoints"
#        if (p1 == p2):
#            assert(False), "cannot make a seg from identical points"
        self.p1 = p1
        self.p2 = p2
        self.isVert = (self.kind() == "vert")
        self.isHoriz = (self.kind() == "horiz")

    def __eq__(self, other):
        if ((self.p1 == other.p1) and (self.p2 == other.p2)):
            return True
        elif ((self.p1 == other.p2) and (self.p2 == other.p1)): 
            return True
        else:
            return False

    def __str__(self):
        return "(%f,%f)-(%f,%f)" % (self.p1.x, self.p1.y,
                                   self.p2.x, self.p2.y)

    def __repr__(self):
        return "Seg("+repr(self.p1)+","+repr(self.p2)+")"

    def __hash__(self):
        hashables = (self.p1.x, self.p1.y, self.p2.x, self.p2.y)
        return hash(hashables)

    def kind(self):
        if (self.p1.x == self.p2.x):
            return "vert"
        elif (self.p1.y == self.p2.y):
            return "horiz"
        else:
            return "other"


class Ray(object):
    def __init__(self, eye, target):
        if (eye == target):
            assert(False), "cannot make a ray from identical points"
        self.eye = eye
        self.dx = target.x - eye.x
        self.dy = target.y - eye.y

    def __eq__(self, other):
        return ((self.eye == other.eye) and (self.dx == other.dx)
            and (self.dy == other.dy))

    def dot(self, other):
        if (type(other) != Ray):
            assert(False), "cannot dot non-rays"
        if (self.eye != other.eye):
            assert(False), "rays have different starting points"
        return self.dx * other.dx + self.dy * other.dy

    def norm(self):
        return math.sqrt(self.dot(self))

    def angle(self, other):
        if (type(other) != Ray):
            assert(False), "angle not defined for non-rays"
        if (self.eye != other.eye):
            assert(False), "rays have different starting points"
        angle = math.acos(self.dot(other) / float(self.norm() * other.norm()))
        return angle # radians


class Intersection(object):
    # represents an intersection of a ray and a wall, which can be either 
    # "normal" - ray.eye --- ray.target --- wall
    #   this should usually obscure the wall
    # "behind" - ray.eye --- wall --- ray.target
    #   the wall is generally in front of the obstruction
    # "backwards" - wall --- ray.eye --- ray.target
    #   this generally does not obscure the wall
    # "infinity"
    #   the ray and wall are parallel
    def __init__(self, point, kind):
        self.point = point
        self.kind = kind # str

    def __eq__(self, other):
        return ((self.kind == other.kind) and (self.point == other.point))

    def __str__(self):
        if (self.kind == "infinity"):
            if (self.point.x == 0):
                return "(0,"+str(sign(self.point.y))+"inf)"
            elif (self.point.y == 0):
                return "("+str(sign(self.point.x))+"inf,0)"
        else:
            return "(%f,%f)" % (self.point.x, self.point.y)

    def __repr__(self):
        return "Intersection(%s,%s)" % (str(repr(self.point)), self.kind)

        


################################################################################
##### Line Intersection Functions ##############################################
################################################################################



def intersectRayAndRookSeg(ray, segment):
    """Given a ray and a rook segment, returns their intersection.  The point
    of intersection is not guaranteed to lie on the segment."""
    if (segment.isHoriz):
        return intersectRayAndHorizSegment(ray, segment)
    elif (segment.isVert):
        return intersectRayAndVertSegment(ray, segment)
    else:
        assert(False), "not a rook segment"

def intersectRayAndVertSegment(ray, segment):
    """Given a ray and a "vert" segment, return an intersection, unless
    the ray and segment are collinear, at which point this will return
    the entire segment"""
    ### NOTE:  The "eye" is forbidden from lying on a segment ###
    # sanity check
    if (not segment.isVert):
        assert(False), "not a vertical segment"
    if (ray.dx != 0):
        # Note that segment.p1.x == segment.p2.x since it is vertical
        # pointOnLine = k*(dx,dy) + eye.  This solves for k:
        k = (segment.p1.x - ray.eye.x) / float(ray.dx)
        yIntercept = k*ray.dy + ray.eye.y
        intPoint = Point(segment.p1.x, yIntercept)
        if (k < 0):
            return Intersection(intPoint, "backwards")
        elif (k < 1):
            return Intersection(intPoint, "behind")
        else:
            return Intersection(intPoint, "normal")
    else:
        if (segment.p1.x == ray.eye.x):
            # collinear!
            return segment
        else:
            yIntercept = 1 if (ray.dy > 0) else -1
            return Intersection(Point(0,yIntercept), "infinity")



def intersectRayAndHorizSegment(ray, segment):
    """Given a ray and a "horiz" segment, return an intersection, unless
    the ray and segment are collinear, at which point this will return
    the entire segment"""
    # sanity check
    if (not segment.isHoriz):
        assert(False), "not a horizontal segment"
    if (ray.dy != 0):
        # Note that segment.p1.y == segment.p2.y since it is horizontal
        # pointOnLine = k*(dx,dy) + eye.  This solves for k:
        k = (segment.p1.y - ray.eye.y) / float(ray.dy)
        xIntercept = k*ray.dx + ray.eye.x
        intPoint = Point(xIntercept, segment.p1.y)
        if (k < 0):
            return Intersection(intPoint, "backwards")
        elif (k < 1):
            return Intersection(intPoint, "behind")
        else:
            return Intersection(intPoint, "normal")
    else:
        if (segment.p1.y == ray.eye.y):
            # collinear!
            return segment
        else:
            xIntercept = 1 if (ray.dx > 0) else -1
            return Intersection(Point(xIntercept,0), "infinity")

def intersectWalls(seg1, seg2):
    """Given two orthogonal rook segs, returns the predicted
    intersection (if they were to stretch into lines)"""
    if (seg1.kind() == seg2.kind()):
        assert(False), "segs not perpendicular"
    if ((seg1.kind() == "other") or (seg2.kind() == "other")):
        assert(False), "not rook segments"
    elif (seg1.isHoriz):
        return Point(seg2.p1.x, seg1.p1.y)
    elif (seg1.isVert):
        return Point(seg1.p1.x, seg2.p1.y)
    else:
        # should never happen
        assert(False), "ERROR!"


################################################################################
##### Line Intersection Functions ##############################################
################################################################################


def obstructViaIntersections(cross1, cross2, wall, seg):
    """Given two intersections, a wall, and a segment, return a set containing the
    portions on the segment.  The wall is what obscured the segment to produce
    the two intersections (which are collinear with the seg)."""
    # sanity check
    if ((type(cross1) != Intersection) or (type(cross2) != Intersection)):
        assert(False), "received non-intersections"
    elif (type(seg) != Seg):
        assert(False), "received non-segment"
    # I recognize this is AWFUL style, but I can't think of another
    #  way to handle these cases
    # Most of these cases are really distinct
    if (cross1.kind == "normal"):
        if (cross2.kind == "normal"):
            return normNormIntersect(cross1,cross2,wall,seg)
        elif (cross2.kind == "behind"):
            return normBehindIntersect(cross1,cross2,wall,seg)
        elif (cross2.kind == "backwards"):
            return normBackIntersect(cross1,cross2,wall,seg)
        elif (cross2.kind == "infinity"):
            return normInfIntersect(cross1,cross2,wall,seg)
    elif (cross1.kind == "behind"):
        if (cross2.kind == "normal"):
            return normBehindIntersect(cross2,cross1,wall,seg)
        elif (cross2.kind == "behind"):
            return behindBehindIntersect(cross1,cross2,wall,seg)
        elif (cross2.kind == "backwards"):
            return behindBackIntersect(cross1,cross2,wall,seg)
        elif (cross2.kind == "infinity"):
            return behindInfIntersect(cross1,cross2,wall,seg)
    elif (cross1.kind == "backwards"):
        if (cross2.kind == "normal"):
            return normBackIntersect(cross2,cross1,wall,seg)
        elif (cross2.kind == "behind"):
            return behindBackIntersect(cross2,cross1,wall,seg)
        elif (cross2.kind == "backwards"):
            return backBackIntersect(cross1,cross2,wall,seg)
        elif (cross2.kind == "infinity"):
            return backInfIntersect(cross1,cross2,wall,seg)
    elif (cross1.kind == "infinity"):
        if (cross2.kind == "normal"):
            return normInfIntersect(cross2,cross1,wall,seg)
        elif (cross2.kind == "behind"):
            return behindInfIntersect(cross2,cross1,wall,seg)
        elif (cross2.kind == "backwards"):
            return backInfIntersect(cross2,cross1,wall,seg)
        elif (cross2.kind == "infinity"):
            return infInfIntersect(cross1,cross2,wall,seg)

def normNormIntersect(cross1,cross2,wall,seg):
    # sanity check
    if ((type(cross1) != Intersection) or (type(cross2) != Intersection)):
        assert(False), "received non-intersections"
    if (type(seg) != Seg):
        assert(False), "received non-seg"
    if (seg.isVert):
        return normNormVertIntersect(cross1,cross2,seg)
    elif (seg.isHoriz):
        return normNormHorizIntersect(cross1,cross2,seg)

def normNormHorizIntersect(cross1,cross2,seg):
    crossPoint1 = cross1.point
    crossPoint2 = cross2.point
    segSet = set([seg.p1, seg.p2])
    crossSet = set([crossPoint1, crossPoint2])
    (minSegPoint, maxSegPoint) = extremeX(segSet)
    (minCrossPoint, maxCrossPoint) = extremeX(crossSet)
    if (minCrossPoint.x <= minSegPoint.x):
        if (maxCrossPoint.x < minSegPoint.x):
            # nothing obscured
            #print "nothing obscured"
            return set([seg])
        elif (maxCrossPoint.x < maxSegPoint.x):
            # obscured on left
            #print "obscured on left"
            return set([Seg(maxCrossPoint, maxSegPoint)])
        else:
            # entirely obscured
            #print "entirely obscured"
            return set()
    elif (minCrossPoint.x < maxSegPoint.x):
        if (maxCrossPoint.x < maxSegPoint.x):
            # centrally obscured
            #print "centrally obscured"
            return set([Seg(minSegPoint,minCrossPoint),
                        Seg(maxCrossPoint,maxSegPoint)])
        else:
            # obscured on right
            #print "obscured on right"
            return set([Seg(minSegPoint,minCrossPoint)])
    else:
        return set([seg])

def normNormVertIntersect(cross1,cross2,seg):
    #print "VERT"
    crossPoint1 = cross1.point
    crossPoint2 = cross2.point
    segSet = set([seg.p1, seg.p2])
    crossSet = set([crossPoint1, crossPoint2])
    (minSegPoint, maxSegPoint) = extremeY(segSet)
    (minCrossPoint, maxCrossPoint) = extremeY(crossSet)
    #print "minCrossPt, minSegPt",minCrossPoint, minSegPoint
    if (minCrossPoint.y <= minSegPoint.y):
        if (maxCrossPoint.y < minSegPoint.y):
            # nothing obscured
            #print "nothing obscured"
            return set([seg])
        elif (maxCrossPoint.y < maxSegPoint.y):
            # obscured on top
            #print "obscured on top"
            return set([Seg(maxCrossPoint, maxSegPoint)])
        else:
            # entirely obscured
            #print "entirely obscured"
            return set()
    elif (minCrossPoint.y < maxSegPoint.y):
        if (maxCrossPoint.y < maxSegPoint.y):
            # centrally obscured
            #print "centrally obscured"
            return set([Seg(minSegPoint,minCrossPoint),
                        Seg(maxCrossPoint,maxSegPoint)])
        else:
            # obscured on bottom
            #print "obscured on bottom"
            return set([Seg(minSegPoint,minCrossPoint)])
    else:
        return set([seg])

def normBehindIntersect(cross,behindCross,wall,seg):
    newCross = intersectWalls(wall,seg)
    newIntersection = Intersection(newCross, "normal")
    #print newIntersection, cross
    return normNormIntersect(cross, newIntersection, wall, seg)

def normBackIntersect(cross,backCross,wall,seg):
    # we want to find the remaining portion of the seg
    #  on the opposite side of the backCross
    if (seg.isVert):
        return normBackVertIntersect(cross,backCross,wall,seg)
    elif (seg.isHoriz):
        return normBackHorizIntersect(cross,backCross,wall,seg)
    else:
        assert(False), "seg should be vert or horiz"

def normBackVertIntersect(normCross, backCross, wall, seg):
    cross = intersectWalls(wall, seg)
    segSet = set([seg.p1, seg.p2])
    (minSegPoint, maxSegPoint) = extremeY(segSet)
    if (backCross.point.y < cross.y):
        # want bottom half of line
        botPoint = extremeY(set([minSegPoint, normCross.point]))[0] # min
        topPoint = extremeY(set([maxSegPoint, normCross.point]))[0] # min
        if (topPoint.y < botPoint.y):
            return set()
        else:
            return set([Seg(botPoint, topPoint)])
    else:
        # want top half of line
        botPoint = extremeY(set([minSegPoint, normCross.point]))[1] # max
        topPoint = extremeY(set([maxSegPoint, normCross.point]))[1] # max
        if (topPoint.y < botPoint.y):
            return set()
        else:
            return set([Seg(botPoint, topPoint)])

def normBackHorizIntersect(normCross, backCross, wall, seg):
    cross = intersectWalls(wall, seg)
    segSet = set([seg.p1, seg.p2])
    (minSegPoint, maxSegPoint) = extremeX(segSet)
    if (backCross.point.x < cross.x):
        # want left half of line
        leftPoint = extremeX(set([minSegPoint, normCross.point]))[0] # min
        rightPoint = extremeX(set([maxSegPoint, normCross.point]))[0] # min
        if (rightPoint.x <= leftPoint.x):
            return set()
        else:
            return set([Seg(leftPoint, rightPoint)])
    else:
        # want right half of line
        leftPoint = extremeX(set([minSegPoint, normCross.point]))[1] # max
        rightPoint = extremeX(set([maxSegPoint, normCross.point]))[1] # max
        if (rightPoint.x <= leftPoint.x):
            return set()
        else:
            return set([Seg(leftPoint, rightPoint)])
    

def normInfIntersect(cross,infCross,wall,seg):
    # we want to find the remaining portion of the seg
    #  on the opposite side of the infCross
    if (seg.isVert):
        return normInfVertIntersect(cross,infCross,wall,seg)
    elif (seg.isHoriz):
        return normInfHorizIntersect(cross,infCross,wall,seg)
    else:
        assert(False), "seg should be vert or horiz"

def normInfVertIntersect(cross, infCross, wall, seg):
    segSet = set([seg.p1, seg.p2])
    (minSegPoint, maxSegPoint) = extremeY(segSet)
    if (infCross.point.y > 0):
        # obscured above cross
        topPoint = extremeY(set([maxSegPoint, cross.point]))[0] # min
        botPoint = extremeY(set([minSegPoint, cross.point]))[0] # min
        if (topPoint.y < botPoint.y):
            return set()
        else:
            return set([Seg(botPoint, topPoint)])
    elif (infCross.point.y < 0):
        # obscured below cross
        topPoint = extremeY(set([maxSegPoint, cross.point]))[1] # max
        botPoint = extremeY(set([minSegPoint, cross.point]))[1] # max
        if (topPoint.y < botPoint.y):
            return set()
        else:
            return set([Seg(botPoint, topPoint)])
    else:
        assert(False), "infCross should be vertical"


def normInfHorizIntersect(cross, infCross, wall, seg):
    segSet = set([seg.p1, seg.p2])
    (minSegPoint, maxSegPoint) = extremeX(segSet)
    if (infCross.point.x > 0):
        # obscured to right of cross
        rightPoint = extremeX(set([maxSegPoint, cross.point]))[0] # min
        leftPoint = extremeX(set([minSegPoint, cross.point]))[0] # min
        if (rightPoint.x < leftPoint.x):
            return set()
        else:
            return set([Seg(leftPoint, rightPoint)])
    elif (infCross.point.x < 0):
        # obscured to left of cross
        rightPoint = extremeX(set([maxSegPoint, cross.point]))[1] # max
        leftPoint = extremeX(set([minSegPoint, cross.point]))[1] # max
        if (rightPoint.x < leftPoint.x):
            return set()
        else:
            return set([Seg(leftPoint, rightPoint)])
    else:
        assert(False), "infCross should be horizontal"

def behindBehindIntersect(behindCross1,behindCross2,wall,seg):
    # the (obstructing) wall is behind the seg
    # so nothing is obstructed
    return set([seg])

def behindBackIntersect(behindCross,backCross,wall,seg):
    # requires a picture to understand:
    #  *** ###|
    #  ***.###|
    #  *** ###|
    # if . is the eye and | represents the obstructing wall:
    #  backCross must be in the * section
    #  behindCross must be in the # section
    # (The eye may not be in a seg)
    # We must remove the part of the seg that extends beyond the wall
    if (seg.isVert):
        return behindBackVertIntersect(behindCross,backCross,wall,seg)
    elif (seg.isHoriz):
        return behindBackHorizIntersect(behindCross,backCross,wall,seg)
    else:
        assert(False), "seg should be vert or horiz"

def behindBackVertIntersect(behindCross, backCross, wall, seg):
    #print "vert"
    newCross = intersectWalls(wall, seg)
    crossSet = set([newCross, behindCross.point, backCross.point])
    (minCrossPoint, maxCrossPoint) = extremeY(crossSet)
    if (wall.p1.y > behindCross.point.y):
        #print "wall crosses above"
        # wall crosses above
        # (could choose backCross, also)
        # quick check
        (botSegPoint,topSegPoint) = extremeY(set([seg.p1,seg.p2]))
        topPoint = extremeY(set([topSegPoint, newCross]))[0] # min
        #print "new seg at", topPoint, botSegPoint
        if (botSegPoint.y >= topPoint.y):
            return set()
        else:
            return set([Seg(botSegPoint, topPoint)])
    else:
        #print "wall crosses below"
        (botSegPoint,topSegPoint) = extremeY(set([seg.p1,seg.p2]))
        botPoint = extremeY(set([botSegPoint, newCross]))[1] # max
        if (topSegPoint.y <= botPoint.y):
            return set()
        else:
            return set([Seg(botPoint, topSegPoint)])


def behindBackHorizIntersect(behindCross, backCross, wall, seg):
    newCross = intersectWalls(wall, seg)
    crossSet = set([newCross, behindCross.point, backCross.point])
    (minCrossPoint, maxCrossPoint) = extremeX(crossSet)
    if (wall.p1.x > behindCross.point.x):
        # wall crosses to right
        # (could choose backCross, also)
        # quick check
        (botSegPoint,topSegPoint) = extremeX(set([seg.p1,seg.p2]))
        topPoint = extremeX(set([topSegPoint, newCross]))[0] # min
        if (botSegPoint.x >= topPoint.x):
            return set()
        else:
            return set([Seg(botSegPoint, topPoint)])
    else:
        (botSegPoint,topSegPoint) = extremeX(set([seg.p1,seg.p2]))
        botPoint = extremeX(set([botSegPoint, newCross]))[1] # max
        if (topSegPoint.x <= botPoint.x):
            return set()
        else:
            return set([Seg(botPoint, topSegPoint)])



def behindInfIntersect(behindCross,infCross,wall,seg):
    # requires a picture:
    #  .###|
    #  *###|
    #  *###|
    # the infCross must be in the * section
    # the behindCross must be in the * section
    # any portion of the segment that extends beyond the wall is obscured
    #  just like with the behindBackIntersect
    if (seg.isVert):
        return behindInfVertIntersect(behindCross,infCross,wall,seg)
    elif (seg.isHoriz):
        return behindInfHorizIntersect(behindCross,infCross,wall,seg)
    else:
        assert(False), "seg should be vert or horiz"


def behindInfVertIntersect(behindCross,infCross,wall,seg):
    segPointSet = set([seg.p1, seg.p2])
    (minSegPoint, maxSegPoint) = extremeY(segPointSet)
    cross = intersectWalls(wall,seg)
    if (infCross.point.y > 0):
        # wall above eye
        topPoint = extremeY(set([maxSegPoint, cross]))[0] # min
        botPoint = extremeY(set([minSegPoint, cross]))[0] # min
        if (topPoint.y < botPoint.y):
            return set()
        else:
            return set([Seg(botPoint, topPoint)])
    elif (infCross.point.y < 0):
        # wall below eye
        topPoint = extremeY(set([maxSegPoint, cross]))[1] # max
        botPoint = extremeY(set([minSegPoint, cross]))[1] # max
        if (topPoint.y < botPoint.y):
            return set()
        else:
            return set([Seg(botPoint, topPoint)])
    else:
        assert(False), "infCross should be vertical"
        

def behindInfHorizIntersect(behindCross,infCross,wall,seg):
    segPointSet = set([seg.p1, seg.p2])
    (minSegPoint, maxSegPoint) = extremeX(segPointSet)
    cross = intersectWalls(wall,seg)
    if (infCross.point.x > 0):
        # wall to right of eye
        leftPoint = extremeX(set([minSegPoint, cross]))[0] # min
        rightPoint = extremeX(set([maxSegPoint, cross]))[0] # min
        if (rightPoint.x < leftPoint.x):
            return set()
        else:
            return set([Seg(leftPoint, rightPoint)])
    elif (infCross.point.x < 0):
        # wall to left of eye
        leftPoint = extremeX(set([minSegPoint, cross]))[1] # max
        rightPoint = extremeX(set([maxSegPoint, cross]))[1] # max
        if (rightPoint.x < leftPoint.x):
            return set()
        else:
            return set([Seg(leftPoint, rightPoint)])
    else:
        assert(False), "infCross should be horizontal"

def backBackIntersect(backCross1,backCross2,wall,seg):
    return set([seg])
            
def backInfIntersect(backCross,infCross,wall,seg):
    return set([seg])

def infInfIntersect(infCross1,infCross2,wall,seg):
    # eye is collinear with wall, which is parallel to seg
    return set([seg])


################################################################################
##### Total Visibility of a Segment ############################################
################################################################################

def obstructSeg(eye, wall, seg):
    """Given an eye, a certain seg, and an (obstructing) wall, this returns
    the remaining visible portion of the seg as a set of segments (or an empty
    set)."""
    ray1 = Ray(eye, wall.p1)
    ray2 = Ray(eye, wall.p2)
    cross1 = intersectRayAndRookSeg(ray1, seg)
    cross2 = intersectRayAndRookSeg(ray2, seg)
    if ((type(cross1) == Seg) or (type(cross2) == Seg)):
        # something obscured entire segment
        # NOTE: There is a small side effect, since
        # the entire seg is returned even if the obstruction lies behind
        # however, the seg must be viewed straight on, so in the 3D case,
        # it is equivalent
        return set()
    #print "\t\tCurrently Obstructing", seg
    #print "\t\twith wall", wall
    #print "\t\tAt intersections", cross1.point, cross1.kind
    #print "\t\t             and", cross2.point, cross2.kind
    return obstructViaIntersections(cross1, cross2, wall, seg)

def obstructSegViaSegSet(eye, segSet, seg):
    """Given an eye, a certain seg, and a set of other segs, this returns the
    remaining visible portion of the specific seg when obstructed by the whole
    set."""
    # sanity check
    if (type(seg) != Seg): assert(False), "seg not of type Seg"
    if (type(segSet) != set): assert(False), "segSet not of type set"
    if (type(eye) != Point): assert(False), "eye not a Point"
    remainingPieces = set([seg])
    #print "Now obstructing the seg", seg
    newPieces = set()
    for wall in segSet:
        for piece in remainingPieces:
            #print "\tObstructing the piece", piece
            #print "\tagainst", wall
            p = obstructSeg(eye, wall, piece)
            #print "\tRemainder:",p
            #print "\tthat was the remainder"
            newPieces = newPieces | p # obstructSeg(eye, wall, piece) # union
        remainingPieces = newPieces
        newPieces = set()
    #print "The remaining pieces of were", remainingPieces
    return remainingPieces


def obstructSegs(eye, segSet):
    """Given an eye and a set of segments, this returns the visible portions
    (as a set) of each segment."""
    visible = set()
    for seg in segSet:
        otherSegs = copy.copy(segSet)
        otherSegs.remove(seg)
        visible = visible.union(obstructSegViaSegSet(eye, otherSegs, seg))
    return visible


################################################################################
##### Camera Class #############################################################
################################################################################

class Camera(object):
    def __init__(self, startPoint, startDir):
        pass
    pass

################################################################################
##### Maze Class ###############################################################
################################################################################

class Maze(object):
    def __init__(self, rows, cols):
        (self.rows, self.cols) = (rows, cols)
        self.initCells()
        self.initPoints()
        self.initSegs()
        self.initCamera()
        self.makeMaze()

    def initCells(self):
        (rows, cols) = (self.rows, self.cols)
        # more points than cells
        cRows = rows - 1
        cCols = cols - 1
        self.cells = [[i+cRows*j for i in xrange(cRows)] for j in xrange(cCols)]

    def initCellsAsOne(self):
        (rows, cols) = (self.rows, self.cols)
        # more points than cells
        cRows = rows - 1
        cCols = cols - 1
        self.cells = [[1]*cCols for i in xrange(cRows)]

    def initPoints(self):
        (rows, cols) = (self.rows, self.cols)
        self.points = [[0]*cols for i in xrange(rows)]
        for row in xrange(rows):
            for col in xrange(cols):
                self.points[row][col] = Point(row, col)

    def initSegs(self):
        # we start with all possible segments
        (rows, cols) = (self.rows, self.cols)
        self.segs = list()
        for row in xrange(rows):
            for col in xrange(cols):
                if (row + 1 < rows):
                    nextPoint = Point(row+1,col)
                    self.segs.append(Seg(Point(row,col), nextPoint))
                if (col + 1 < cols):
                    nextPoint = Point(row,col+1)
                    self.segs.append(Seg(Point(row,col), nextPoint))

    def initCamera(self):
        # we start closest to (0,0)
        startPoint = Point(0.5, 0.5)
        startDir = Ray(startPoint, Point(1,0))
        self.camera = Camera(startPoint, startDir)

    def removeSeg(self, seg, cellVal1, cellVal2):
        if (seg in self.segs):
            self.segs.remove(seg)
        self.renameCells(cellVal1, cellVal2)

    def renameCells(self, cellVal1, cellVal2):
        (cRows, cCols) = (self.rows - 1, self.cols - 1)
        (fromVal, toVal) = (max(cellVal1, cellVal2), min(cellVal1, cellVal2))
        for row in xrange(cRows):
            for col in xrange(cCols):
                if (self.cells[row][col] == fromVal):
                    self.cells[row][col] = toVal

    def isFinishedMaze(self):
        (cRows, cCols) = (self.rows - 1, self.cols - 1)
        for row in xrange(cRows):
            for col in xrange(cCols):
                if (self.cells[row][col] != 0):
                    return False
        return True
            

    def makeMaze(self):
        # I am borrowing heavily from the algorithm used here:
        # kosbie.net/cmu/fall-12/15-112/handouts/notes-recursion/mazeSolver.py
        (rows, cols) = (self.rows, self.cols)
        (cRows, cCols) = (rows-1, cols-1)
        #print "(cRows,cCols) = ", (cRows, cCols)
        while (not self.isFinishedMaze()):
            cRow = random.randint(0, cRows-1)
            cCol = random.randint(0, cCols-1)
            #print "\t",(cRow,cCol)
            curCell = self.cells[cRow][cCol]
            if flipCoin(): # try to go east
                if (cCol == cCols - 1): continue # at edge
                targetCell = self.cells[cRow][cCol + 1]
                dividingSeg = Seg(Point(cRow,cCol+1),
                                  Point(cRow+1,cCol+1))
                if (curCell == targetCell):
                    if (dividingSeg in self.segs):
                        if (smallChance()):
                            print "YES"
                            self.removeSeg(dividingSeg, curCell, targetCell)
                        else:
                            print "OH WELL"
                        #continue
                    #else:# True: # NOTE: NOTE: NOTE:smallChance():
                    #    self.removeSeg(dividingSeg, curCell, targetCell)
                    #else:
                    #    continue
                else:
                    self.removeSeg(dividingSeg, curCell, targetCell)
            else: # try to go north
                if (cRow == cRows - 1): continue # at edge
                targetCell = self.cells[cRow+1][cCol]
                dividingSeg = Seg(Point(cRow+1,cCol),
                                  Point(cRow+1,cCol+1))
                if (curCell == targetCell):
                    continue
                else:
                    self.removeSeg(dividingSeg, curCell, targetCell)
            #print self.cells
            #time.sleep(0.5)
            #print self.isFinishedMaze()

    def deadCornerCell(self, row, col, dir):
        #print (row,col)
        (rows, cols) = (self.rows, self.cols)
        (cRows, cCols) = (rows - 1, cols - 1)
        if (dir == "UL"):
            # checking to the upper left
            # if shielded by dead cells to the bottom right, this is dead
            rightCell = self.cells[row][col+1]
            downCell = self.cells[row-1][col]
            return (((self.hasSeg(row, col, "right")) or (rightCell == 0)) and
                    ((self.hasSeg(row, col, "down")) or (downCell == 0)))
        elif (dir == "UR"):
            leftCell = self.cells[row][col-1]
            downCell = self.cells[row-1][col]
            return (((self.hasSeg(row, col, "left")) or (leftCell == 0)) and
                    ((self.hasSeg(row, col, "down")) or (downCell == 0)))
        elif (dir == "DL"):
            rightCell = self.cells[row][col+1]
            upCell = self.cells[row+1][col]
            return (((self.hasSeg(row, col, "right")) or (rightCell == 0)) and
                    ((self.hasSeg(row, col, "up")) or (upCell == 0)))
        elif (dir == "DR"):
            leftCell = self.cells[row][col-1]
            upCell = self.cells[row+1][col]
            return (((self.hasSeg(row, col, "left")) or (leftCell == 0)) and
                    ((self.hasSeg(row, col, "up")) or (upCell == 0)))
        else:
            assert(False), "not a direction"



        return False

    def cullCorners(self, eye):
        eyeRow = int(math.floor(eye.y))
        eyeCol = int(math.floor(eye.x))
        (rows, cols) = (self.rows, self.cols)
        (cRows, cCols) = (rows - 1, cols - 1)
        # xranges are reversed so that we check progressively
        # further from the eye (since this process "cascades")
        culledFlag = False
        # bottom left
        if ((eyeRow != 0) and (eyeCol != 0)):
            for row in xrange(eyeRow-1, -1, -1):
                for col in xrange(eyeCol-1, -1, -1):
                    if (self.deadCornerCell(row, col, "DL")):
                        #print "DL-DEAD ->", (row,col)
                        #print "DL"
                        if (self.cells[row][col] != 0):
                            self.cells[row][col] = 0 # dead
                            culledFlag = True
        # bottom right
        if ((eyeRow != 0) and (eyeCol != cCols)):
            for row in xrange(eyeRow-1, -1, -1):
                for col in xrange(eyeCol+1, cCols):
                    if (self.deadCornerCell(row, col, "DR")):
                        #print "DR-DEAD ->", (row,col)
                        #print "DR"
                        if (self.cells[row][col] != 0):
                            self.cells[row][col] = 0 # dead
                            culledFlag = True
        # top left
        if ((eyeRow != cRows) and (eyeCol != 0)):
            #print "HEY, IN UL NOW"
            #print "eyeRow, cRows", eyeRow, cRows
            #print "eyeCol", eyeCol
            for row in xrange(eyeRow+1, cRows):
                #print "row", row
                for col in xrange(eyeCol-1, -1, -1):
                    #print "col"
                    #print "UL row,col",(row,col)
                    if (self.deadCornerCell(row, col, "UL")):
                        #print "UL-DEAD ->", (row,col)
                        #print "UL"
                        if (self.cells[row][col] != 0):
                            self.cells[row][col] = 0 # dead
                            culledFlag = True
        # top right
        if ((eyeRow != cRows) and (eyeCol != cCols)):
            for row in xrange(eyeRow+1, cRows):
                for col in xrange(eyeCol+1, cCols):
                    if (self.deadCornerCell(row, col, "UR")):
                        #print "UR-DEAD ->", (row,col)
                        #print "UR"
                        if (self.cells[row][col] != 0):
                            self.cells[row][col] = 0 # dead
                            culledFlag = True
        return culledFlag # something was deleted
                

        
    def removeDeadSandwichedSegs(self):
        (rows, cols) = (self.rows, self.cols)
        (cRows, cCols) = (rows - 1, cols - 1)
        # check right
        for row in xrange(cRows):
            for col in xrange(cCols - 1):
                if (self.cells[row][col] == self.cells[row][col+1] == 0):
                    deadSeg = Seg(Point(col+1, row), Point(col+1, row+1))
                    if (deadSeg in self.checkSegs):
                        #print "REMOVED",deadSeg
                        self.checkSegs.remove(deadSeg)
        # check far right
        for row in xrange(cRows):
            if (self.cells[row][cCols-1] == 0):
                deadSeg = Seg(Point(cCols+1, row), Point(cCols+1, row+1))
                if (deadSeg in self.checkSegs):
                    #print "FAR RIGHT"
                    self.checkSegs.remove(deadSeg)
        # check up
        for row in xrange(cRows - 1):
            for col in xrange(cCols):
                if (self.cells[row][col] == self.cells[row+1][col] == 0):
                    deadSeg = Seg(Point(col, row+1), Point(col+1, row+1))
                    if (deadSeg in self.checkSegs):
                        #print "REMOVED",deadSeg
                        self.checkSegs.remove(deadSeg)
        # check far top
        for col in xrange(cCols):
            if (self.cells[cRows-1][col] == 0):
                deadSeg = Seg(Point(col, cRows+1), Point(col+1, cRows+1))
                if (deadSeg in self.checkSegs):
                    #print "FAR TOP"
                    self.checkSegs.remove(deadSeg)
        return None
                        
                    

    def hasSeg(self, row, col, dir):
        y = row
        x = col
        if (dir == "left"):
            return (Seg(Point(x, y), Point(x, y+1)) in self.checkSegs)
        elif (dir == "right"):
            return (Seg(Point(x+1,y), Point(x+1,y+1)) in self.checkSegs)
        elif (dir == "up"):
            return (Seg(Point(x, y+1), Point(x+1, y+1)) in self.checkSegs)
        elif (dir == "down"):
            return (Seg(Point(x, y), Point(x+1, y)) in self.checkSegs)
        else:
            assert(False), "not a direction"

    def deleteCellsInDir(self, delRow, delCol, dir):
        # destructive function
        (rows, cols) = (self.rows, self.cols)
        (cRows, cCols) = (rows - 1, cols - 1)
        #print "delRow, delCol",(delRow, delCol)
        #print "cRows, cCols", (cRows, cCols)
        if ((delRow == cRows) or (delRow < 0) or
            (delCol == cCols) or (delCol < 0)):
            # out of bounds
        #    print "OUT OF BOUNDS"
            return None
        if (dir == "left"):
        #    print "... too theeee lleeffttt"
            for col in xrange(0, delCol+1):
        #        print "left -> ", delRow, col
                self.cells[delRow][col] = 0
        #    print self.cells
        elif (dir == "right"):
        #    print "... too theeee rriiggthhtt"
            for col in xrange(delCol, cCols):
        #        print "right -> ", delRow, col
                self.cells[delRow][col] = 0
        #    print self.cells
        elif (dir == "down"):
            for row in xrange(0, delRow+1):
        #        print "down -> ", row, delCol
                self.cells[row][delCol] = 0
        #    print self.cells
        elif (dir == "up"):
            for row in xrange(delRow, cRows):
        #        print "up -> ", row, delCol
                self.cells[row][delCol] = 0
        #    print self.cells
        else:
            assert(False), "not a direction"


    def cullSegs(self, eye):
        # only return segs which could possibly be visible to reduce 
        # render time
        # mark all cells as 1 (alive)
        # we will mark cells as 0 (dead) if they cannot possible be seen
        # walls sandwiched between dead cells are invisible and will be culled
        eyeRow = int(math.floor(eye.y))
        eyeCol = int(math.floor(eye.x))
        (rows, cols) = (self.rows, self.cols)
        (cRows, cCols) = (rows - 1, cols - 1)
        self.initCellsAsOne()
        self.checkSegs = copy.copy(self.segs)
        #print self.cells
        for col in xrange(eyeCol, cCols):
            if self.hasSeg(eyeRow, col, "right"):
                #print "RIGHT"
                self.deleteCellsInDir(eyeRow, col+1, "right")
                break
        for col in xrange(eyeCol, -1, -1):
            if self.hasSeg(eyeRow, col, "left"):
                #print "LEFT"
                self.deleteCellsInDir(eyeRow, col-1, "left")
                break
        for row in xrange(eyeRow, cRows):
            if self.hasSeg(row, eyeCol, "up"):
                #print "UP"
                self.deleteCellsInDir(row+1, eyeCol, "up")
                break
        for row in xrange(eyeRow, -1, -1):
            if self.hasSeg(row, eyeCol, "down"):
                #print "DOWN"
                self.deleteCellsInDir(row-1, eyeCol, "down")
                break
        while(self.cullCorners(eye)):
            # cullCorners will remove cells invisible by a corner
            # it will return true if something was removed
            pass
        self.removeDeadSandwichedSegs()
        # will remove segs sandwiched between dead cells
        #print "\n\n####################################"
        #print self.cells
        #print eye
        return set(self.checkSegs)


        







################################################################################
##### Animation Class ##########################################################
################################################################################


class Animation(object):
    def __init__(self, width=500, height=300):
        self.root = Tk()
        self.width = width
        self.height = height
        self.canvas = Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack()
        self.init()
        self.root.bind("<KeyPress>", self.keyPressed)
        self.root.bind("<KeyRelease>", self.keyReleased)
        self.root.bind("<Button-1>", self.mousePressed)

    def run(self):
        self.timerFired()
        self.root.mainloop()

    def init(self):
        pass

    def redrawAll(self):
        pass

    def keyPressed(self, event):
        pass

    def keyReleased(self, event):
        pass

    def mousePressed(self, event):
        pass

    def timerFired(self):
        self.redrawAll()
        delay = 100 # ms
        self.canvas.after(delay, self.timerFired)


################################################################################
##### MazeGame Animation Class #################################################
################################################################################


# TODO: Move animation functions into MazeGame class
class MazeGame(Animation):
    def __init__(self):
        pass

    def init(self):
        pass

def run():
    global canvas
    root = Tk()
    canvas = Canvas(root, width=700, height=700)
    canvas.pack()
    class Struct: pass
    canvas.data = Struct()
    init()
    root.bind("<KeyPress>", keyPressed)
    root.bind("<KeyRelease>", keyReleased)
    timerFired()
    root.mainloop()

def init():
    canvas.data.counter = 1
    canvas.eye = Point(0.5,0.5)
    canvas.maze = Maze(12,12)
    canvas.segs = set(canvas.maze.segs)
    canvas.v = (0,0)
#    canvas.segs = set([Seg(Point(4,1),Point(4,5)),
#                       Seg(Point(3,1),Point(3,3)),
#                       #Seg(Point(4,1),Point(4,5)),
#                       Seg(Point(2,3),Point(2,5)),
#                       Seg(Point(2,6),Point(8,6)),
#                       Seg(Point(4,7),Point(3,7)),
#                       Seg(Point(3,4),Point(3,7)),
#                       Seg(Point(8,1),Point(8,5)),
#                       Seg(Point(4,5),Point(6,5)),
#                       Seg(Point(7,3),Point(7,5)),
#                       Seg(Point(6,2),Point(6,5)),
#                       Seg(Point(1,4),Point(5,4))])
    
def timerFired():
    canvas.eye = Point(canvas.eye.x + canvas.v[0], canvas.eye.y + canvas.v[1])
    redrawAll()
    delay = 40 # ms
    canvas.data.counter += 1
    #print canvas.data.counter
    canvas.after(delay, timerFired)

# TODO: have 3D (with glasses), 3D (without), and top-down drawing modes
def redrawAll():
    canvas.delete(ALL)
    eye = canvas.eye
    segs = canvas.segs
    canvas.create_line(5+50*eye.x-1, 5+50*eye.y-1, 5+50*eye.x+1, 5+50*eye.y+1, fill="red", width=2)
    for seg in segs:
        canvas.create_line(5+50*seg.p1.x, 5+50*seg.p1.y, 5+50*seg.p2.x, 5+50*seg.p2.y)
    colors = ["red"]
    canvas.maze.initCellsAsOne()
    possibleSegs = canvas.maze.cullSegs(eye)
    #possibleSegs = segs
    #print "########################################"
    #print "########################################"
    #print possibleSegs
    #print "########################################"
    #print segs
    #print "########################################"
    for s in possibleSegs:
        canvas.create_line(5+50*s.p1.x, 5+50*s.p1.y, 5+50*s.p2.x, 5+50*s.p2.y,
                           fill=colors[0], width=3)
    visible = obstructSegs(eye, possibleSegs)
    #print "visible = ",visible
#    for s in visible:
#        canvas.create_line(5+50*s.p1.x, 5+50*s.p1.y, 5+50*s.p2.x, 5+50*s.p2.y,
#                           fill=colors[0], width=3)

def keyPressed(event):
    if (event.keysym == "Up"):
        canvas.v = (0,-0.1)
    elif (event.keysym == "Down"):
        canvas.v = (0,0.1)
    elif (event.keysym == "Left"):
        canvas.v = (-0.1,0)
    elif (event.keysym == "Right"):
        canvas.v = (0.1,0)
    #redrawAll()
    #print """
    #########################################################
    ####EYE = """,canvas.eye,"""
    #########################################################"""

def keyReleased(event):
    canvas.v = (0,0)
    #redrawAll()



run()




