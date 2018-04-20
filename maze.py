#!/bin/env pypy

# 3D-maze
# maze.py


# Copyright 2014, Jackson Bahr

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import copy
import math
from Tkinter import *
import random
import time

CYCLE_AMOUNT = 5 # higher number -> fewer cycles

# if you change the resolution,
# you may need to slightly alter the following two:
CAM_HEIGHT = 0.125
CAM_WIDTH = 0.015

# do not edit these:
CAM_LENGTH = 0.1
CAM_SEP = 0.04
WALL_H = 0.5
CELL_SIZE = 40 # pixels
DEBUG = False
FOV = math.pi/2

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
    return random.choice(choices)

def withinEp(x, y):
    """Returns True if x and y are within some predefined epsilon: 0.0001"""
    epsilon = 0.0001
    return abs(x-y) < epsilon

def chopDomain(x):
    """Chops number to fit within [1,1] for use with arccos"""
    if (x > 1):
        return 1
    elif (x < -1):
        return -1
    else:
        return x

def sec(x):
    return 1 / math.cos(x)

def csc(x):
    return 1 / math.sin(x)

def isNumber(x):
    """Returns True if x in an int, long, or float; False otherwise"""
    return type(x) in (int, long, float)

def sign(x):
    """Returns "+" for non-negative numbers; "-" otherwise"""

def mathSign(x):
    if (x == 0):
        return 0
    else:
        return (x/abs(x))


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

# from http://www.kosbie.net/cmu/spring-14/15-112/handouts/grayScale.py
def hexColor(red, green, blue):
    return ("#%02x%02x%02x" % (red, green, blue))

# color gradient idea from Delancey Wu
def makeColor(row, col, rows, cols):
    if (((row == rows-1) and (col == cols-2)) or
          ((row == rows-2) and (col == cols-1))):
        color = hexColor(255,255,255)
    else:
        green = 255*(row+col)/float(rows+cols)
        blue = 255*(rows+cols-row-col)/float(rows+cols)
        red = 40
        color = hexColor(red, green, blue)
    return color

def rgbFromHex(color):
    # get RGB color from hex
    red = int(color[1:3], 16)
    green = int(color[3:5], 16)
    blue = int(color[5:], 16)
    return (red, green, blue)

def rightChannelColor(color):
    # red-tint color for left channel of
    # red-cyan anaglyph
#    (red, green, blue) = rgbFromHex(color)
#    gray = (red+green+blue)/3
#    newGray = min(255, gray*2)
    # I could not get this to work, so I resorted
    # to black/white (rather, red/cyan) for 3DG:
    return hexColor(255, 0, 0)

def leftChannelColor(color):
    # cyan-tint color for right channel of
    # red-cyan anaglyph
#    (red, green, blue) = rgbFromHex(color)
#    gray = (red+green+blue)/2
#    newGray = min(255, gray*2)
    # I could not get this to work, so I resorted
    # to black/white (rather, red/cyan) for 3DG:
    return hexColor(0, 255, 255)

def shrinkScreenSeg(x, h, otherX, otherH):
    if (abs(x) > abs(h)):
        newX = 1.0*mathSign(x)
        a = newX / x
        newH = a*h + (1-a)*otherH
    else:
        newH = 1.0*mathSign(h)
        a = newH / h
        newX = a*x + (1-a)*otherX
    return (newX, newH)



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

    def dist(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx**2 + dy**2)

class Seg(object):
    def __init__(self, p1, p2, color=hexColor(0,0,0)):
        # sanity check
        if not (type(p1) == type(p2) == Point):
            assert(False), "cannot make a seg from nonpoints"
        self.p1 = p1
        self.p2 = p2
        self.color = color
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

    def withinDist(self, eye, dist):
        """Returns True if eye is within a certain distance of
        the segment (in a rectangular sense around the edges)"""
        minX = min(self.p1.x, self.p2.x)
        maxX = max(self.p1.x, self.p2.x)
        minY = min(self.p1.y, self.p2.y)
        maxY = max(self.p1.y, self.p2.y)
        return ((minX - dist < eye.x < maxX + dist) and
                (minY - dist < eye.y < maxY + dist))


class Ray(object):
    def __init__(self, eye, target):
        self.eye = eye
        self.dx = target.x - eye.x
        self.dy = target.y - eye.y
        self.target = target

    def __eq__(self, other):
        return ((self.eye == other.eye) and (self.dx == other.dx)
            and (self.dy == other.dy))

    def __str__(self):
        return str(self.eye) + " -> " + str(self.target)

    def __repr__(self):
        return "Ray("+repr(self.eye)+","+repr(self.target)+")"

    def __sub__(self, other):
        if (self.eye != other.eye):
            assert(False), "cannot subtract"
        return Ray(other.target, self.target)

    def __mul__(self, scale):
        newTargetX = self.eye.x + scale*self.dx
        newTargetY = self.eye.y + scale*self.dy
        return Ray(self.eye, Point(newTargetX, newTargetY))

    def __rmul__(self, scale):
        return self.__mul__(scale)


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
        value = chopDomain(self.dot(other) /float(self.norm() * other.norm()))
        angle = math.acos(value)
        return angle # radians

    def rotate(self, angle):
        # angle in radians
        # taken from jbahr-hw8.py
        rotationMatrix = Matrix([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]])
        oldNorm = self.norm()
        dir = Vector([self.dx, self.dy])
        newDir = rotationMatrix.mult(dir)
        newPoint = Point(newDir.elements[0], newDir.elements[1])
        newTarget = Point(newPoint.x + self.eye.x, newPoint.y + self.eye.y)
        newRay = Ray(self.eye, newTarget)
        return newRay

    def angleWithX(self):
        xAxis = Ray(self.eye, Point(self.eye.x + 1, self.eye.y))
        return self.angle(xAxis)
        

class Matrix(object):
    def __init__(self, elements):
        self.elements = elements
        self.rows = len(elements)
        if (type(elements[0]) == list):
            self.cols = len(elements[0])
        else:
            # handle vectors too
            self.cols = 1

    def mult(self, other):
        if (type(other) != Vector):
            assert(False), "cannot be multiplied"
        # other must be a vector
        def dotprod(row):
            return Vector(row).dot(other)
        # using map was the idea of James Wu 
        return Vector(map(dotprod, self.elements))

    def transpose(self):
        # this implementation was taken from recitation
        return Matrix(zip(*self.elements))

    def __str__(self):
        output = ""
        for i in xrange(self.rows):
            for j in xrange(self.cols):
                output += str(self.elements[i][j])
            output += "\n"
        return output

    def __repr__(self):
        return "Matrix("+str(self.elements)+")"

class Vector(Matrix):
    def __init__(self, elements):
        # column vector
        self.elements = elements
        self.rows = len(elements)

    def dot(self, other):
        if (type(other) != Vector):
            assert(False), "cannot dot non-vectors"
        def prod(a):
            return a[0] * a[1]
        return sum(map(prod,zip(self.elements,other.elements)))

    def norm(self):
        return math.sqrt(self.dot(self))

    def __mul__(self, other):
        if (not isNumber(other)):
            assert(False), "cannot multiply"
        return Vector(map(lambda x : other*x, self.elements))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        newElements = map(sum, zip(self.elements, other.elements))
        return Vector(newElements)

    def __neg__(self):
        return Vector(map(lambda x: -x, self.elements))

    def __str__(self):
        return str(self.elements)

class ScreenSeg(object):
    # keeps track of distance from center of screen and height
    def __init__(self, cam, seg):
        self.color = seg.color
        # MUST be given a seg that is visible in the direction of the cam
        # Seg pruning must be done beforehand!
        # following is some linear algebra
        v1 = Ray(cam.viewRay.eye, seg.p1)
        v2 = Ray(cam.viewRay.eye, seg.p2)

        screenV1 = v1 * (cam.viewRay.norm()**2 / cam.viewRay.dot(v1))
        screenV2 = v2 * (cam.viewRay.norm()**2 / cam.viewRay.dot(v2))

        self.x1 = screenV1.dot(cam.rightRay)
        self.x2 = screenV2.dot(cam.rightRay)

        self.h1 = WALL_H * screenV1.norm() / v1.norm()
        self.h2 = WALL_H * screenV2.norm() / v2.norm()

        self.shrink()

    def __str__(self):
        (x1,h1,x2,h2) = (self.x1,self.h1,self.x2,self.h2)
        return str((x1,h1,x2,h2))

    def shrink(self):
        # required because tkinter cannot draw points too far from
        # the visible portion of the canvas
        # if abs(x),abs(h) < 1, it can be drawn
        (x1, x2, h1, h2) = (self.x1, self.x2, self.h1, self.h2)
        if ((abs(x1) > 1) or (abs(h1) > 1)):
            (self.x1, self.h1) = shrinkScreenSeg(x1, h1, x2, h2)
        if ((x2 > 1) or (h2 > 1)):
            (self.x2, self.h2) = shrinkScreenSeg(x2, h2, x1, h1)


        

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
            return set([seg])
        elif (maxCrossPoint.x < maxSegPoint.x):
            # obscured on left
            return set([Seg(maxCrossPoint, maxSegPoint, seg.color)])
        else:
            # entirely obscured
            return set()
    elif (minCrossPoint.x < maxSegPoint.x):
        if (maxCrossPoint.x < maxSegPoint.x):
            # centrally obscured
            return set([Seg(minSegPoint,minCrossPoint, seg.color),
                        Seg(maxCrossPoint,maxSegPoint, seg.color)])
        else:
            # obscured on right
            return set([Seg(minSegPoint,minCrossPoint, seg.color)])
    else:
        return set([seg])

def normNormVertIntersect(cross1,cross2,seg):
    crossPoint1 = cross1.point
    crossPoint2 = cross2.point
    segSet = set([seg.p1, seg.p2])
    crossSet = set([crossPoint1, crossPoint2])
    (minSegPoint, maxSegPoint) = extremeY(segSet)
    (minCrossPoint, maxCrossPoint) = extremeY(crossSet)
    if (minCrossPoint.y <= minSegPoint.y):
        if (maxCrossPoint.y < minSegPoint.y):
            # nothing obscured
            return set([seg])
        elif (maxCrossPoint.y < maxSegPoint.y):
            # obscured on top
            return set([Seg(maxCrossPoint, maxSegPoint, seg.color)])
        else:
            # entirely obscured
            return set()
    elif (minCrossPoint.y < maxSegPoint.y):
        if (maxCrossPoint.y < maxSegPoint.y):
            # centrally obscured
            return set([Seg(minSegPoint,minCrossPoint, seg.color),
                        Seg(maxCrossPoint,maxSegPoint, seg.color)])
        else:
            # obscured on bottom
            return set([Seg(minSegPoint,minCrossPoint, seg.color)])
    else:
        return set([seg])

def normBehindIntersect(cross,behindCross,wall,seg):
    newCross = intersectWalls(wall,seg)
    newIntersection = Intersection(newCross, "normal")
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
            return set([Seg(botPoint, topPoint, seg.color)])
    else:
        # want top half of line
        botPoint = extremeY(set([minSegPoint, normCross.point]))[1] # max
        topPoint = extremeY(set([maxSegPoint, normCross.point]))[1] # max
        if (topPoint.y < botPoint.y):
            return set()
        else:
            return set([Seg(botPoint, topPoint, seg.color)])

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
            return set([Seg(leftPoint, rightPoint, seg.color)])
    else:
        # want right half of line
        leftPoint = extremeX(set([minSegPoint, normCross.point]))[1] # max
        rightPoint = extremeX(set([maxSegPoint, normCross.point]))[1] # max
        if (rightPoint.x <= leftPoint.x):
            return set()
        else:
            return set([Seg(leftPoint, rightPoint, seg.color)])
    

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
            return set([Seg(botPoint, topPoint, seg.color)])
    elif (infCross.point.y < 0):
        # obscured below cross
        topPoint = extremeY(set([maxSegPoint, cross.point]))[1] # max
        botPoint = extremeY(set([minSegPoint, cross.point]))[1] # max
        if (topPoint.y < botPoint.y):
            return set()
        else:
            return set([Seg(botPoint, topPoint, seg.color)])
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
            return set([Seg(leftPoint, rightPoint, seg.color)])
    elif (infCross.point.x < 0):
        # obscured to left of cross
        rightPoint = extremeX(set([maxSegPoint, cross.point]))[1] # max
        leftPoint = extremeX(set([minSegPoint, cross.point]))[1] # max
        if (rightPoint.x < leftPoint.x):
            return set()
        else:
            return set([Seg(leftPoint, rightPoint, seg.color)])
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
    newCross = intersectWalls(wall, seg)
    crossSet = set([newCross, behindCross.point, backCross.point])
    (minCrossPoint, maxCrossPoint) = extremeY(crossSet)
    if (wall.p1.y > behindCross.point.y):
        # wall crosses above
        # (could choose backCross, also)
        # quick check
        (botSegPoint,topSegPoint) = extremeY(set([seg.p1,seg.p2]))
        topPoint = extremeY(set([topSegPoint, newCross]))[0] # min
        if (botSegPoint.y >= topPoint.y):
            return set()
        else:
            return set([Seg(botSegPoint, topPoint, seg.color)])
    else:
        (botSegPoint,topSegPoint) = extremeY(set([seg.p1,seg.p2]))
        botPoint = extremeY(set([botSegPoint, newCross]))[1] # max
        if (topSegPoint.y <= botPoint.y):
            return set()
        else:
            return set([Seg(botPoint, topSegPoint, seg.color)])


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
            return set([Seg(botSegPoint, topPoint, seg.color)])
    else:
        (botSegPoint,topSegPoint) = extremeX(set([seg.p1,seg.p2]))
        botPoint = extremeX(set([botSegPoint, newCross]))[1] # max
        if (topSegPoint.x <= botPoint.x):
            return set()
        else:
            return set([Seg(botPoint, topSegPoint, seg.color)])



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
            return set([Seg(botPoint, topPoint, seg.color)])
    elif (infCross.point.y < 0):
        # wall below eye
        topPoint = extremeY(set([maxSegPoint, cross]))[1] # max
        botPoint = extremeY(set([minSegPoint, cross]))[1] # max
        if (topPoint.y < botPoint.y):
            return set()
        else:
            return set([Seg(botPoint, topPoint, seg.color)])
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
            return set([Seg(leftPoint, rightPoint, seg.color)])
    elif (infCross.point.x < 0):
        # wall to left of eye
        leftPoint = extremeX(set([minSegPoint, cross]))[1] # max
        rightPoint = extremeX(set([maxSegPoint, cross]))[1] # max
        if (rightPoint.x < leftPoint.x):
            return set()
        else:
            return set([Seg(leftPoint, rightPoint, seg.color)])
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
        # however, the seg must be viewed straight on for this to happen
        #, so in the 3D case, it doesn't matter
        return set()
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
    newPieces = set()
    for wall in segSet:
        for piece in remainingPieces:
            newPieces = newPieces | obstructSeg(eye, wall, piece) # union
        remainingPieces = newPieces
        newPieces = set()
    return remainingPieces


def obstructSegs(eye, segSet):
    """Given an eye and a set of segments, this returns the visible portions
    (as a set) of each segment."""
    visible = set()
    for seg in segSet:
        otherSegs = segSet - set([seg])
        visible = visible.union(obstructSegViaSegSet(eye, otherSegs, seg))
    return visible


################################################################################
##### Camera Class #############################################################
################################################################################

class Camera(object):
    def __init__(self, viewRay):
        if (type(viewRay) != Ray):
            assert(False), "Camera requires Ray"
        self.viewRay = viewRay
        self.rightRay = viewRay.rotate(-math.pi/2)
        self.height = CAM_HEIGHT

    def rotate(self, angle):
        # angle in radians
        viewRay = self.viewRay
        self.viewRay = self.viewRay.rotate(angle)
        self.rightRay = self.rightRay.rotate(angle)

    def translate(self, vector):
        newX = self.viewRay.eye.x + vector.elements[0]
        newY = self.viewRay.eye.y + vector.elements[1]
        newEye = Point(newX, newY)
        newTargetX = self.viewRay.target.x + vector.elements[0]
        newTargetY = self.viewRay.target.y + vector.elements[1]
        newTarget = Point(newTargetX, newTargetY)
        newRightX = self.rightRay.target.x + vector.elements[0]
        newRightY = self.rightRay.target.y + vector.elements[1]
        newRightTarget = Point(newRightX, newRightY)
        self.viewRay = Ray(newEye, newTarget)
        self.rightRay = Ray(newEye, newRightTarget)
        

################################################################################
##### Maze Class ###############################################################
################################################################################

class Maze(object):
    def __init__(self, rows, cols):
        (self.rows, self.cols) = (rows, cols)
        self.initCells()
        self.initPoints()
        self.initSegs()
        self.makeMaze()

    def initCells(self):
        (rows, cols) = (self.rows, self.cols)
        # more points than cells
        cRows = rows - 1
        cCols = cols - 1
        self.cells = [[i+cCols*j for i in xrange(cCols)] for j in xrange(cRows)]

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
                curPoint = Point(row,col)
                color = makeColor(row, col, rows, cols)
                if (row + 1 < rows):
                    nextPoint = Point(row+1,col)
                    self.segs.append(Seg(curPoint, nextPoint, color))
                if (col + 1 < cols):
                    nextPoint = Point(row,col+1)
                    self.segs.append(Seg(curPoint, nextPoint, color))

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
        while (not self.isFinishedMaze()):
            cRow = random.randint(0, cRows-1)
            cCol = random.randint(0, cCols-1)
            curCell = self.cells[cRow][cCol]
            if flipCoin(): # try to go east
                if (cCol == cCols - 1): continue # at edge
                targetCell = self.cells[cRow][cCol + 1]
                dividingSeg = Seg(Point(cRow,cCol+1),
                                  Point(cRow+1,cCol+1))
                if (curCell == targetCell):
                    if (dividingSeg in self.segs):
                        if (smallChance()):
                            self.removeSeg(dividingSeg, curCell, targetCell)
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

    def deadCornerCell(self, row, col, dir):
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
                        if (self.cells[row][col] != 0):
                            self.cells[row][col] = 0 # dead
                            culledFlag = True
        # bottom right
        if ((eyeRow != 0) and (eyeCol != cCols)):
            for row in xrange(eyeRow-1, -1, -1):
                for col in xrange(eyeCol+1, cCols):
                    if (self.deadCornerCell(row, col, "DR")):
                        if (self.cells[row][col] != 0):
                            self.cells[row][col] = 0 # dead
                            culledFlag = True
        # top left
        if ((eyeRow != cRows) and (eyeCol != 0)):
            for row in xrange(eyeRow+1, cRows):
                for col in xrange(eyeCol-1, -1, -1):
                    if (self.deadCornerCell(row, col, "UL")):
                        if (self.cells[row][col] != 0):
                            self.cells[row][col] = 0 # dead
                            culledFlag = True
        # top right
        if ((eyeRow != cRows) and (eyeCol != cCols)):
            for row in xrange(eyeRow+1, cRows):
                for col in xrange(eyeCol+1, cCols):
                    if (self.deadCornerCell(row, col, "UR")):
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
                        self.checkSegs.remove(deadSeg)
        # check far right
        for row in xrange(cRows):
            if (self.cells[row][cCols-1] == 0):
                deadSeg = Seg(Point(cCols+1, row), Point(cCols+1, row+1))
                if (deadSeg in self.checkSegs):
                    self.checkSegs.remove(deadSeg)
        # check up
        for row in xrange(cRows - 1):
            for col in xrange(cCols):
                if (self.cells[row][col] == self.cells[row+1][col] == 0):
                    deadSeg = Seg(Point(col, row+1), Point(col+1, row+1))
                    if (deadSeg in self.checkSegs):
                        self.checkSegs.remove(deadSeg)
        # check far top
        for col in xrange(cCols):
            if (self.cells[cRows-1][col] == 0):
                deadSeg = Seg(Point(col, cRows+1), Point(col+1, cRows+1))
                if (deadSeg in self.checkSegs):
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
        if ((delRow == cRows) or (delRow < 0) or
            (delCol == cCols) or (delCol < 0)):
            # out of bounds
            return None
        if (dir == "left"):
            for col in xrange(0, delCol+1):
                self.cells[delRow][col] = 0
        elif (dir == "right"):
            for col in xrange(delCol, cCols):
                self.cells[delRow][col] = 0
        elif (dir == "down"):
            for row in xrange(0, delRow+1):
                self.cells[row][delCol] = 0
        elif (dir == "up"):
            for row in xrange(delRow, cRows):
                self.cells[row][delCol] = 0
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
        for col in xrange(eyeCol, cCols):
            if self.hasSeg(eyeRow, col, "right"):
                self.deleteCellsInDir(eyeRow, col+1, "right")
                break
        for col in xrange(eyeCol, -1, -1):
            if self.hasSeg(eyeRow, col, "left"):
                self.deleteCellsInDir(eyeRow, col-1, "left")
                break
        for row in xrange(eyeRow, cRows):
            if self.hasSeg(row, eyeCol, "up"):
                self.deleteCellsInDir(row+1, eyeCol, "up")
                break
        for row in xrange(eyeRow, -1, -1):
            if self.hasSeg(row, eyeCol, "down"):
                self.deleteCellsInDir(row-1, eyeCol, "down")
                break
        while(self.cullCorners(eye)):
            # cullCorners will remove cells invisible by a corner
            # it will return true if something was removed
            pass
        self.removeDeadSandwichedSegs()
        # will remove segs sandwiched between dead cells
        return set(self.checkSegs)


        







################################################################################
##### Animation Class ##########################################################
################################################################################


# taken from jbahr-hw9.py
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


class MazeGame(Animation):
    def __init__(self, mazeSize, width=700, height=500):
        self.mazeRows = mazeSize
        self.mazeCols = mazeSize
        self.mode = "3D"
        super(MazeGame, self).__init__(width, height)
        self.root.resizable(width=0, height=0) # non-resizable

######################
####### Model ########
######################

    def init(self):
        self.isGameOver = False
        self.isHelp = True
        self.initMaze()
        self.initCamera()

    def initCamera(self):
        self.speed = 0.08
        self.rotateSpeed = math.pi/32
        self.cameraLength = CAM_LENGTH
        self.cameraSep = CAM_SEP
        # we start closest to (0,0)
        # facing in the x direction
        # check if facing wall
        startPoint = Point(0.5, 0.5)
        if (Seg(Point(0,1),Point(1,1)) in self.maze.segs):
            secondCamStart = Point(0.5, 0.5 - self.cameraSep)
            secondCamView = Point(0.5 + self.cameraLength, 0.5 - self.cameraSep)
            viewPoint = Point(0.5 + self.cameraLength, 0.5)
        else:
            secondCamStart = Point(0.5 + self.cameraSep, 0.5)
            secondCamView = Point(0.5 + self.cameraSep, 0.5 + self.cameraLength)
            viewPoint = Point(0.5, 0.5 + self.cameraLength)
        self.camera = Camera(Ray(startPoint, viewPoint))
        self.secondCamera = Camera(Ray(secondCamStart, secondCamView))
        self.cameraVel = 0
        self.sideCameraVel = 0
        self.cameraRotVel = 0

    def initMaze(self):
        print "Generating random maze..."
        self.maze = Maze(self.mazeRows, self.mazeCols)
        print "Finished generating maze!"


######################
##### Controller #####
######################

    def timerFired(self):
        if (self.mode == "3D"):
            self.screenSegs = set()
            self.visibleSegs = set()
            self.circularVisibleSegs = set()
            (self.visibleSegs,
             self.circularVisibleSegs) = self.firstPersonVisibleSegs(self.camera)
            self.projectVisibleSegsToScreen(self.camera,
                                            self.screenSegs,
                                            self.visibleSegs)
        elif (self.mode == "3DG"):
            self.screenSegs = set()
            self.secScreenSegs = set()
            self.visibleSegs = set()
            self.secVisibleSegs = set()
            self.circularVisibleSegs = set()
            self.secCircularVisibleSegs = set()
            (self.visibleSegs, self.circularVisibleSegs) = \
                                        self.firstPersonVisibleSegs(self.camera)
            (self.secVisibleSegs, self.secCircularVisibleSegs) = \
                    self.firstPersonVisibleSegs(self.secondCamera)
            self.projectVisibleSegsToScreen(self.camera,
                                            self.screenSegs,
                                            self.visibleSegs)
            self.projectVisibleSegsToScreen(self.secondCamera,
                                            self.secScreenSegs,
                                            self.secVisibleSegs)
        elif (self.mode == "2D"):
            self.topDownVisibleSegs()
        self.updateCamera()
        self.isWin()
        self.redrawAll()
        delay = 40 # ms
        self.canvas.after(delay, self.timerFired)

    def topDownVisibleSegs(self):
        eye = self.camera.viewRay.eye
        possibleSegs = self.maze.cullSegs(eye)
        self.circularVisibleSegs = obstructSegs(eye, possibleSegs)

    def firstPersonVisibleSegs(self, cam):
        # check if each seg in visibleSegs is within 90 degrees of cam.viewRay
        # given visSegs to store visibleSegs and circSegs to store
        # circularly visible segs
        visSegs = set()
        circSegs = set()
        eye = cam.viewRay.eye
        possibleSegs = self.maze.cullSegs(eye)
        circSegs = obstructSegs(eye, possibleSegs) # visible in 360
        visSegs = set()
        for seg in circSegs:
            ray1 = Ray(eye, seg.p1)
            ray2 = Ray(eye, seg.p2)
            angle1 = abs(ray1.angle(cam.viewRay))
            angle2 = abs(ray2.angle(cam.viewRay))
            screenEye = cam.viewRay.target
            screenDir = (cam.rightRay.dx, cam.rightRay.dy)
            rightPoint = Point(screenEye.x+screenDir[0], screenEye.y+screenDir[1])
            screenRay = Ray(screenEye, rightPoint)
            viewRay = cam.viewRay
            if ((angle1 < FOV) and (angle2 < FOV)):
                visSegs.add(seg)
            elif ((angle1 >= FOV) and (angle2 < FOV)):
                newIntersect = intersectRayAndRookSeg(screenRay, seg)
                newPoint = newIntersect.point
                # check for special case
                if (0 < ray2.dot(viewRay)/viewRay.norm() < viewRay.norm()):
                    continue
                else:
                    visSegs.add(Seg(newPoint, seg.p2, seg.color))
            elif ((angle1 < FOV) and (angle2 >= FOV)):
                newIntersect = intersectRayAndRookSeg(screenRay, seg)
                newPoint = newIntersect.point
                # check for special case
                if (0 < ray1.dot(viewRay)/viewRay.norm() < viewRay.norm()):
                    continue
                else:
                    visSegs.add(Seg(seg.p1, newPoint, seg.color))
            else:
                # seg is completely behind
                continue
        return (visSegs, circSegs)
                
    def projectVisibleSegsToScreen(self, cam, screenSegs, visSegs):
        for seg in visSegs:
            screenSegs.add(ScreenSeg(cam, seg))

    def updateCamera(self):
        topDownModes = ["2D"]
        firstPersonModes = ["3D", "3DG"]
        if (self.mode in topDownModes):
            self.topDownUpdateCamera()
        elif (self.mode in firstPersonModes):
            self.firstPersonUpdateCamera()
        else:
            assert(False), "Not a valid mode"


    def topDownUpdateCamera(self):
        self.camera.rotate(self.cameraRotVel)
        self.secondCamera.rotate(self.cameraRotVel)
        self.camera.translate(self.cameraVel)
        self.secondCamera.translate(self.cameraVel)
        if not self.cameraIsLegal():
            # move back!
            self.camera.translate(- self.cameraVel)
            self.secondCamera.translate(- self.cameraVel)


    def firstPersonUpdateCamera(self):
        viewDir = Vector([self.camera.viewRay.dx, self.camera.viewRay.dy])
        rightDir = Vector([self.camera.rightRay.dx, self.camera.rightRay.dy])
        velocity = (self.cameraVel/viewDir.norm()) * viewDir
        sideVel = (self.sideCameraVel/rightDir.norm()) * rightDir
        newVel = (velocity + sideVel)
        if (newVel.norm() != 0):
            oldNorm = max(velocity.norm(), sideVel.norm())
            newVel = newVel * (oldNorm/newVel.norm())
        self.camera.rotate(self.cameraRotVel)
        self.secondCamera.rotate(self.cameraRotVel)
        self.camera.translate(newVel)
        self.secondCamera.translate(newVel)
        if not self.cameraIsLegal():
            # move back!
            self.camera.translate(- newVel)
            self.secondCamera.translate(- newVel)


    def cameraIsLegal(self):
        # check that camera is no more than 1.5*self.cameraLength
        # from any wall
        cam1 = self.camera
        cam2 = self.secondCamera
        for seg in self.circularVisibleSegs:
            if (seg.withinDist(cam1.viewRay.eye,1.2*self.cameraLength) or
                (seg.withinDist(cam2.viewRay.eye, 1.2*self.cameraLength))):
                return False
        return True

    def isWin(self):
        # if in last cell, game is won
        lastX = self.maze.cols - 1
        lastY = self.maze.rows - 1
        if ((abs(self.camera.viewRay.eye.x - lastX) < 1) and
            (abs(self.camera.viewRay.eye.y - lastY) < 1)):
            self.isGameOver = True

    def mousePressed(self, event):
        pass

    def keyPressed(self, event):
        firstPersonModes = ["3D", "3DG"]
        topDownModes = ["2D"]
        if (self.mode in firstPersonModes):
            self.firstPersonKeyPressed(event)
        elif (self.mode in topDownModes):
            self.topDownKeyPressed(event)
        else:
            assert(False), "not a valid mode"
        if (event.keysym == "h"):
            # toggle help screen
            self.isHelp = not self.isHelp
        else:
            self.isHelp = False
        if (event.keysym == "r"):
            self.mode = "3D"
            self.init()
            self.isHelp = False
            # restart
        if (event.keysym == "1"):
            self.mode = "2D"
            self.cameraVel = Vector([0,0])
        elif (event.keysym == "2"):
            self.mode = "3D"
            self.cameraVel = 0
            self.sideCameraVel = 0
        elif (event.keysym == "3"):
            self.mode = "3DG"
            self.cameraVel = 0
            self.sideCameraVel = 0


    def firstPersonKeyPressed(self, event):
        viewDir = Vector([self.camera.viewRay.dx, self.camera.viewRay.dy])
        if ((event.keysym == "w") or (event.keysym=="Up")):
            self.cameraVel = self.speed 
        elif ((event.keysym == "s") or (event.keysym=="Down")):
            self.cameraVel = -self.speed
        elif ((event.keysym == "d") or (event.keysym=="Right")):
            # clockwise
            self.cameraRotVel = self.rotateSpeed
        elif ((event.keysym == "a") or (event.keysym=="Left")):
            # counter-clockwise
            self.cameraRotVel = - self.rotateSpeed
        elif ((event.keysym == "comma") or (event.keysym=="Prior")):
            # Prior is page up
            # sidestep left 
            self.sideCameraVel = self.speed
        elif ((event.keysym == "period") or (event.keysym=="Next")):
            # Next is page down
            self.sideCameraVel = - self.speed

    def topDownKeyPressed(self, event):
        up = ["Up", "w"]
        down = ["Down", "s"]
        left = ["Left", "a"]
        right = ["Right", "d"]
        if (event.keysym in up):
            # up is down in TkInter
            self.cameraVel = self.speed * Vector([0,-1])
        elif (event.keysym in down):
            # up is down in TkInter
            self.cameraVel = self.speed * Vector([0,1])
        elif (event.keysym in right):
            self.cameraVel = self.speed * Vector([1,0])
        elif (event.keysym in left):
            self.cameraVel = self.speed * Vector([-1,0])
        # ensure no more rotation
        self.cameraRotVel = 0

    def keyReleased(self, event):
        firstPersonModes = ["3D", "3DG"]
        topDownModes = ["2D"]
        if (self.mode in firstPersonModes):
            return self.firstPersonKeyReleased(event)
        elif (self.mode in topDownModes):
            return self.topDownKeyReleased(event)
        else:
            assert(False), "not a valid mode"

    def firstPersonKeyReleased(self, event):
        translations = ["w", "s", "Up", "Down"]
        sideSteps = ["comma","period","Prior","Next"]
        rotations = ["a", "d", "Left", "Right"]
        if (event.keysym in translations):
            self.cameraVel = 0
        elif (event.keysym in sideSteps):
            self.sideCameraVel = 0
        elif (event.keysym in rotations):
            self.cameraRotVel = 0

    def topDownKeyReleased(self, event):
        translations = ["Up", "Down", "Left", "Right",
                        "w", "a", "s", "d"]
        if (event.keysym in translations):
            self.cameraVel = Vector([0,0])


######################
######## View ########
######################

    def drawGameOver(self):
        cx = self.width/2
        cy = self.height/2
        self.canvas.create_text(cx, cy, text="You Win!",
                                font="Helvetica 36 bold")

    def draw3DGGameOver(self):
        # If you find yourself on a system where Tkinter text
        # stippling works, then comment out:
        self.drawGameOver()
        # and uncomment the remainder of this function 
        # (and install the dejavu font)
#        cxLeft = self.width * (49.5/100)
#        cxRight = self.width * (50.5/100)
#        cy = self.height/2
#        self.canvas.create_text(cxLeft, cy, text="You Win!",
#                                font=("dejavu sans light", 36),
#                                stipple="gray50",
#                                fill=None,
#                                offset="0,0")
#        self.canvas.create_text(cxRight, cy, text="You Win!",
#                                font=("dejavu sans light", 36),
#                                stipple="gray50",
#                                fill=hexColor(0,255,255),
#                                offset="0,1")

    def drawHelp(self):
        cx = self.width/2
        leftcx = self.width/3
        rightcx = (2*self.width)/3
        cy = self.height/2
        self.canvas.create_text(cx,cy*(2./9.),text="3D Maze!",
                                font="Helvetica 28",fill="white")
        self.canvas.create_text(cx,cy/2,
                                text="""Find the far corner (the white cell)
The maze will get greener""",
                                font="Helvetica 24",fill="white",
                                justify=CENTER)
        self.canvas.create_text(leftcx, cy, text="""
        To move
        To sidestep
        To switch to 2D mode
                           3D mode
                           3D with glasses
        To toggle help
        To restart""", font="Helvetica 18",fill="white")
        self.canvas.create_text(rightcx, cy, text="""
        WASD or arrow keys
        ,/. or PgUp/PgDown
        1
        2
        3
        h
        r""", font="Helvetica 18",fill="white")

        


    def redrawAll(self):
        self.canvas.delete(ALL)
        if (self.mode == "2D"):
            self.redraw2D()
        elif (self.mode == "3D"):
            self.redraw3D()
        elif (self.mode == "3DG"):
            self.redraw3DG()
        else:
            assert(False), "no valid mode"
        if (self.isHelp):
            self.drawHelp()
        elif (self.isGameOver):
            if (self.mode == "3DG"):
                self.draw3DGGameOver()
            else:
                self.drawGameOver()

    def redraw2D(self):
        eye = self.camera.viewRay.eye
        segs = self.circularVisibleSegs
        cx = self.width/2
        cy = self.height/2
        left = cx - (CELL_SIZE*(self.mazeCols - 1))/2
        top = cy - (CELL_SIZE*(self.mazeRows - 1))/2
        self.draw2DEye(left, top)
        for s in segs:
            self.canvas.create_line(left + CELL_SIZE*s.p1.x,
                                    top + CELL_SIZE*s.p1.y,
                                    left + CELL_SIZE*s.p2.x,
                                    top + CELL_SIZE*s.p2.y,
                                    fill=s.color, width=2)
        if (DEBUG):
            for s in self.maze.segs:
                self.canvas.create_line(left + CELL_SIZE*s.p1.x,
                                        top + CELL_SIZE*s.p1.y,
                                        left + CELL_SIZE*s.p2.x,
                                        top + CELL_SIZE*s.p2.y,
                                        fill="black", width=1)

    def draw2DEye(self, left, top):
        eye = self.camera.viewRay.eye
        target = self.camera.viewRay.target
        x1 = left + CELL_SIZE*eye.x
        y1 = top + CELL_SIZE*eye.y
        x2 = left + CELL_SIZE*target.x
        y2 = top + CELL_SIZE*target.y
        self.canvas.create_line(x1, y1, x2, y2, arrow="last", fill="blue")

    def drawBackground(self):
        background = hexColor(255,255,255) # better for red/cyan anaglyph
        self.canvas.create_rectangle(0, 0, self.width, self.height,
                                     fill=background, width=0)

    def drawGround(self):
        brown = hexColor(123, 112, 0)
        #brown = hexColor(163, 130, 41) # alternative
        cy = self.height/2
        self.canvas.create_rectangle(0, cy, self.width, self.height,
                                     fill=brown, width=0)


    def drawSky(self):
        #blue = hexColor(130,202,250) # light sky blue
        blue = hexColor(112,172,255) # sky blue
        #blue = hexColor(160,191,235) # alternative
        cy = self.height/2
        self.canvas.create_rectangle(0, 0, self.width, cy,
                                     fill=blue, width=0)

    def redraw3D(self):
        cx = self.width/2
        cy = self.height/2
        scaleX = (self.width / CAM_WIDTH)
        scaleY = (self.height / CAM_HEIGHT)
        self.drawGround()
        self.drawSky()
        for s in self.screenSegs:
            left = cx - s.x1*scaleX
            right = cx - s.x2*scaleX
            leftTop = cy + s.h1*scaleY
            leftBot = cy - s.h1*scaleY
            rightTop = cy + s.h2*scaleY
            rightBot = cy - s.h2*scaleY
            self.canvas.create_polygon(left, leftTop, right, rightTop,
                                       right, rightBot, left, leftBot,
                                       fill=s.color,
                                       outline="black")

    def draw3DGChannel(self, channel):
        # the wireframe 3DG idea was Nick Goman's
        cx = self.width/2
        cy = self.height/2
        scaleX = (self.width / CAM_WIDTH)
        scaleY = (self.height / CAM_HEIGHT)
        if (channel == "right"):
            screenSegs = self.screenSegs
            shift = "0,0"
        else:
            screenSegs = self.secScreenSegs
            shift = "0,1"
        for s in screenSegs:
            left = cx - s.x1*scaleX
            right = cx - s.x2*scaleX
            leftTop = cy + s.h1*scaleY
            leftBot = cy - s.h1*scaleY
            rightTop = cy + s.h2*scaleY
            rightBot = cy - s.h2*scaleY
            if (channel == "right"):
                outlineColor = hexColor(255,0,0)
            else:
                outlineColor = hexColor(0,255,255)
            if (s.color == "#ffffff"):
                endColor = "#bbffbb"
                self.canvas.create_polygon(left, leftTop, right, rightTop,
                                           right, rightBot, left, leftBot,
                                           width=0, fill=endColor)
            self.canvas.create_line(left, leftTop, right, rightTop,
                                    stipple="gray50", offset=shift,
                                    fill=outlineColor,width=3)
            self.canvas.create_line(right, rightTop, right, rightBot,
                                    stipple="gray50", offset=shift,
                                    fill=outlineColor,width=3)
            self.canvas.create_line(right, rightBot, left, leftBot,
                                    stipple="gray50", offset=shift,
                                    fill=outlineColor,width=3)
            self.canvas.create_line(left, leftBot, left, leftTop,
                                    stipple="gray50", offset=shift,
                                    fill=outlineColor,width=3)


    def redraw3DG(self):
        self.drawBackground() # white
        self.draw3DGChannel("left")
        self.draw3DGChannel("right")

game = MazeGame(10, 1366, 768)
game.run()
