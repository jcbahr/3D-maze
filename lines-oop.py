#!/bin/env python2.7

# lines-oop.py
# line intersection (an object oriented version)

import copy

################################################################################
##### Point & Seg Helper Functions #############################################
################################################################################

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
        return ((self.x == other.x) and (self.y == other.y))

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
        if (p1 == p2):
            assert(False), "cannot make a seg from identical points"
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
        elif (self.p1.y == self.p2py):
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
            if (self.dx == 0):
                return "(0,"+str(sign(self.point.y))+"inf)"
            elif (self.dy == 0):
                return "("+str(sign(self.point.x))+"inf,0)"
        else:
            return "(%f,%f)" % (self.point.x, self.point.y)

    def __repr__(self):
        return "Intersection(%s,%s)" % (str(repr(self.point)), self.kind)

        


################################################################################
##### Line Intersection Functions ##############################################
################################################################################



def intersectRayAndVertSegment(ray, segment):
    """Given a ray and a "vert" segment, return an intersection, unless
    the ray and segment are collinear, at which point this will return
    the entire segment"""
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

def intersectRayAndRookSeg(ray, segment):
    """Given a ray and a rook segment, returns their intersection.  The point
    of intersection is not guaranteed to lie on the segment."""
    if (segment.isHoriz):
        return intersectRayAndHorizSegment(ray, segment)
    elif (segment.isVert):
        return intersectRayAndVertSegment(ray, segment)
    else:
        assert(False), "not a rook segment"

def intersectWalls(seg1, seg2):
    """Given two orthogonal rook segs, returns the predicted
    intersection (if they were to stretch into lines)"""
    if (seg1.kind() == seg2.kind()):
        assert(False), "segs not perpendicular"
    if ((seg1.kind() == "other") or (seg2.kind() == "other")):
        assert(False), "not rook segments"
    elif (seg1.isHoriz):
        return Point(seg1.p1.x, seg2.p1.y)
    elif (seg1.isVert):
        return Point(seg2.p1.x, seg1.p1.y)
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
    if (type(cross1) == "normal"):
        if (type(cross2) == "normal"):
            return normNormIntersect(cross1,cross2,wall,seg)
        elif (type(cross2) == "behind"):
            return normBehindIntersect(cross1,cross2,wall,seg)
        elif (type(cross2) == "backwards"):
            return normBackIntersect(cross1,cross2,wall,seg)
        elif (type(cross2) == "infinity"):
            return normInfIntersect(cross1,cross2,wall,seg)
    elif (type(cross1) == "behind"):
        if (type(cross2) == "normal"):
            return normBehindIntersect(cross2,cross1,wall,seg)
        elif (type(cross2) == "behind"):
            return behindBehindIntersect(cross1,cross2,wall,seg)
        elif (type(cross2) == "backwards"):
            return behindBackIntersect(cross1,cross2,wall,seg)
        elif (type(cross2) == "infinity"):
            return behindInfIntersect(cross1,cross2,wall,seg)
    elif (type(cross1) == "backwards"):
        if (type(cross2) == "normal"):
            return normBackIntersect(cross2,cross1,wall,seg)
        elif (type(cross2) == "behind"):
            return behindBackIntersect(cross2,cross1,wall,seg)
        elif (type(cross2) == "backwards"):
            return backBackIntersect(cross1,cross2,wall,seg)
        elif (type(cross2) == "infinity"):
            return backInfIntersect(cross1,cross2,wall,seg)
    elif (type(cross1) == "infinity"):
        if (type(cross2) == "normal"):
            return normInfIntersect(cross2,cross1,wall,seg)
        elif (type(cross2) == "behind"):
            return behindInfIntersect(cross2,cross1,wall,seg)
        elif (type(cross2) == "backwards"):
            return backInfIntersect(cross2,cross1,wall,seg)
        elif (type(cross2) == "infinity"):
            return infInfIntersect(cross1,cross2,wall,seg)

def normNormIntersect(cross1,cross2,wall,seg):
    # sanity check
    if ((type(cross1) != Intersection) or (type(cross2) != Intersection)):
        assert(False), "received non-intersections"
    if (type(seg) != Seg):
        assert(False), "received non-seg"
    if (seg.isVert):
        return normNormVertIntersection(cross1,cross2,seg)
    elif (seg.isHoriz):
        return normNormHorizIntersection(cross1,cross2,seg)

def normNormHorizIntersect(cross1,cross2,seg):
    crossPoint1 = cross1.point
    crossPoint2 = cross2.point
    segSet = set([seg.p1, seg.p2])
    crossSet = set([crossPoint1, crossPoint2])
    (minSegPoint, maxSegPoint) = extremeX(segSet)
    (minCrossPoint, maxCrossPoint) = extremeX(crossSet)
    if (minCrossPoint.x < minSegPoint.x):
        if (maxCrossPoint.x < minSegPoint.x):
            # nothing obscured
            return set(seg)
        elif (maxCrossPoint.x < maxSegPoint.x):
            # obscured on left
            return set(Seg(maxCrossPoint, maxSegPoint))
        else:
            # entirely obscured
            return set()
    elif (minCrossPoint.x < maxSegPoint.x):
        if (maxCrossPoint.x < maxSegPoint.x):
            # centrally obscured
            return set([Seg(minSegPoint,minCrossPoint),
                        Seg(maxCrossPoint,maxSegPoint)])
        else:
            # obscured on right
            return set(Seg(minSegPoint,minCrossPoint))
    else:
        return set()

def normNormVertIntersection(cross1,cross2,seg):
    crossPoint1 = cross1.point
    crossPoint2 = cross2.point
    segSet = set([seg.p1, seg.p2])
    crossSet = set([crossPoint1, crossPoint2])
    (minSegPoint, maxSegPoint) = extremeY(segSet)
    (minCrossPoint, maxCrossPoint) = extremeY(crossSet)
    if (minCrossPoint.y < minSegPoint.x):
        if (maxCrossPoint.y < minSegPoint.x):
            # nothing obscured
            return set(seg)
        elif (maxCrossPoint.y < maxSegPoint.x):
            # obscured on left
            return set(Seg(maxCrossPoint, maxSegPoint))
        else:
            # entirely obscured
            return set()
    elif (minCrossPoint.y < maxSegPoint.x):
        if (maxCrossPoint.y < maxSegPoint.x):
            # centrally obscured
            return set([Seg(minSegPoint,minCrossPoint),
                        Seg(maxCrossPoint,maxSegPoint)])
        else:
            # obscured on right
            return set(Seg(minSegPoint,minCrossPoint))
    else:
        return set()

def normBehindIntersect(cross,behindCross,wall,seg):
    newCross = intersectWalls(wall,seg)
    return normNormIntersect(cross, newCross, wall, seg)

def normBackIntersect(cross,backCross,wall,seg):
    # we want to find the remaining portion of the seg
    #  on the opposite side of the backCross
    if (seg.isVert):
        return normBackVertIntersection(cross,backCross,wall,seg)
    elif (seg.isHoriz):
        return normBackHorizIntersection(cross,backCross,wall,seg)
    else:
        assert(False), "seg should be vert or horiz"

def normBackVertIntersection(cross, backCross, wall, seg):
    segSet = set([seg.p1, seg.p2])
    (minSegPoint, maxSegPoint) = extremeY(segSet)
    if (backCross.point.y < cross.point.y):
        # want top half of line
        return set(Seg(cross.point, maxSegPoint))
    else:
        return set(Seg(cross.point, minSegPoint))

def normBackHorizIntersection(cross, backCross, wall, seg):
    segSet = set([seg.p1, seg.p2])
    (minSegPoint, maxSegPoint) = extremeX(segSet)
    if (backCross.point.x < cross.point.x):
        # want right half of line
        return set(Seg(cross.point, maxSegPoint))
    else:
        return set(Seg(cross.point, minSegPoint))
    





