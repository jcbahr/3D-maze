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
        # error checking
        if not (type(p1) == type(p2) == Point):
            assert(False), "cannot make a seg from nonpoints"
        if (p1 == p2):
            assert(False), "cannot make a seg from identical points"
        self.p1 = p1
        self.p2 = p2
        self.isVert = (self.kind() == "vert")
        self.isHoriz = (self.kind() == "horiz")

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
    # represents an intersection, which can be either "behind" or "normal"
    def __init__(self, point, kind):
        self.point = point
        self.kind = kind

class InfIntersection(object):
    # "infinity intersection", which is represented as an
    # ordered pair with either +1, -1, or 0 in each spot.  These represent
    # an intersection at infinity"""
    def __init__(self, dx, dy):
        if (dx == dy == 0):
            assert(False), "not an infinity intersection"""
        self.dx = dx
        self.dy = dy

    def __eq__(self, other):
        return ((self.dx == other.dx) and (self.dy == other.dy))

    def __str__(self):
        if (self.dx == 0):
            return "(0,"+str(sign(self.dy))+"inf)"
        elif (self.dy == 0):
            return "("+str(sign(self.dx))+"inf,0)"

    def __repr__(self):
        return "InfIntersection(%d,%d)" % (self.dx, self.dy)



################################################################################
##### Line Intersection Functions ##############################################
################################################################################

def intersectRayAndSegment(ray, segment):
    """Given a ray and a segment, return two intersections"""




