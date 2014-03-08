#!/bin/env python2.7

# a test of line intersection for the term project

################################################################################
##### Point, Line, and Set Helper Functions ####################################
################################################################################

def isNumber(x):
    return ((type(x) == int) or (type(x) == float)
            or (type(x) == long))

def sign(x):
    # zero is considered positive
    # it won't matter for the purposes of this program
    return "+" if (x >= 0) else "-"

def isPoint(p):
    if (type(p) != tuple):
        return False
    elif (len(p) != 2):
        return False
    return isNumber(p[0]) and isNumber(p[1])

def isSeg(seg):
    if (type(seg) != tuple):
        return False
    elif (len(seg) != 2):
        return False
    return isPoint(seg[0]) and isPoint(seg[1])

def makeSegment(p1,p2):
    return (p1,p2)

def getElementFromSet(s):
    # takes an arbitrary element in a set
    #  without modifying the set
    for val in s:
        return val # breaks

def minMaxXPoint(pointSet):
    # return the min/max points of a set in a tuple
    #  with respect to the x coordinate
    # python does this by default
    minPoint = min(pointSet)
    maxPoint = max(pointSet)
    return (minPoint, maxPoint)

def minMaxYPoint(pointSet):
    # return the min/max points of a set in a tuple
    #  with respect to the y coordinate
    # can be remade with "key" of min(), max()
    minPoint = maxPoint = getElementFromSet(pointSet)
    for point in pointSet:
        if (point[1] > maxPoint[1]):
            maxPoint = point
        elif (point[1] < minPoint[1]):
            minPoint = point
    return (minPoint, maxPoint)

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

def intersectRayAndVertSegment(ray, segment):
    ((x1,y1), (x2,y2)) = segment
    (eye, (dx, dy)) = ray
    (eyeX, eyeY) = eye
    xSeg = float(x1) # = x2
    if (dx != 0):
        k = (xSeg - eyeX) / dx # vector form: pointOnLine = k*(dx,dy) + eye
        if (k < 0): # intersects in wrong direction
            return None
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
        if (k < 0): # intersects in wrong direction
            return None
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

#def intersectLineAndVertSegment(line, segment):
#    ((x1,y1), (x2,y2)) = segment
#    (a,b,c) = line
#    yMin = min(y1, y2)
#    yMax = max(y1, y2)
#    if (b != 0):
#        yIntercept = float(c - a*x1)/b
#        intersection = (x1, yIntercept)
#    else: # line itself is vertical
#        if (float(c/a) == float(x1)):
#            # direct line of sight
#            intersection = segment
#        else:
#            intersection = None
#    return intersection
#
#def intersectLineAndHorizSegment(line, segment):
#    ((x1,y1), (x2,y2)) = segment
#    (a,b,c) = line
#    xMin = min(x1, x2)
#    xMax = max(x1, x2)
#    if (a != 0):
#        xIntercept = float(c - b*y1)/a
#        intersection = (xIntercept, y1)
#    else: # line itself is horizontal
#        if (float(c/b) == float(y1)):
#            # direct line of sight
#            intersection = segment
#        else:
#            print "a,b,c:", a,b,c
#            print "y1:", y1
#            print "y2:", y2
#            intersection = None
#    return intersection

#def intersectLineAndRookSegment(line, segment):
#    # should not check if intersection lies on line
#    #  that will be done by the chop functions
#    # segments are a list of two 2-tuples:
#    # rook segments are guaranteed to have either
#    #  x1 = x2 or y1 = y2
#    ((x1,y1), (x2,y2)) = segment
#    (a,b,c) = line
#    if (x1 != x2 and y1 != y2):
#        assert(False), "not a rook segment"
#    segType = "vert" if (x1==x2) else "horiz"
#    if (segType == "vert"):
#        print "vert"
#        return intersectLineAndVertSegment(line, segment)
#    else: # horizontal segment
#        print "horiz"
#        return intersectLineAndHorizSegment(line, segment)


################################################################################
##### Line Obstruction Functions ###############################################
################################################################################

def obstructViaIntersections(intersection, seg):
    obstructions = set()
    ((segX1, segY1), (segX2, segY2)) = seg
    segType = "vert" if (segX1 == segX2) else "horiz"
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
    else:
        print "woah"
    return obstructions


def obstructSegViaSeg(eye, wall, segWithObstructions):
    # "wall" is just shorthand for an obstructing segment
    (seg, obstructions) = segWithObstructions
    (wallP1, wallP2) = wall
    if ((eye == wallP1) or (eye == wallP2)):
        assert(False), "collision with wall edge"
    ray1 = makeRay(eye, wallP1)
    intersection1 = intersectRayAndRookSegment(ray1, seg)
    ray2 = makeRay(eye, wallP2)
    intersection2 = intersectRayAndRookSegment(ray2, seg)
    print "rays:", ray1, ray2
    obstructions = obstructions.union(
                         obstructViaIntersections(intersection1, seg),
                         obstructViaIntersections(intersection2, seg))
    return [seg, obstructions]



#def obstructSegViaSeg(eyePoint, blockingSeg, segWithObstructions):
#    #
#    # WARNING: This function destructively
#    #          modifies the set of obstructions!
#    #
#    # segWithObstructions is a list containing a segment
#    #  and a set of "obstructions", i.e., points on the segment
#    #  which have been obscured by walls
#    (obstructedSeg, obstructions) = segWithObstructions
#    (blockingP1, blockingP2) = blockingSeg
#    print "blocking points:", blockingP1, blockingP2
#    line1 = makeLine(eyePoint, blockingP1)
#    line2 = makeLine(eyePoint, blockingP2)
#    print "lines:",line1, line2
#    intersection1 = intersectLineAndRookSegment(line1, obstructedSeg)
#    intersection2 = intersectLineAndRookSegment(line2, obstructedSeg)
#    if (intersection1 != None):
#        print "intersect1:", intersection1
#        obstructions.add(intersection1)
#    if (intersection2 != None):
#        print "intersect2:", intersection2
#        obstructions.add(intersection2)
#    if (intersection1 == intersection2 == None):
#        # do nothing
#    elif (intersection1 == None): # and int2 != None
#        #### 
#        #####
#        #####
#        ##### WEIRD
#        #####
#        #####
#        ####
#        # want to chop off entire half of line
#    elif (intersection2 == None): # and int1 != None
#        # want to chop off other half of line
#    return [obstructedSeg, obstructions]

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
                                                    [segPiece, set()])
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

        


    # more optimization can be done by only obstructing farther segs
    #  by closer segs, but this is algorithmically difficult for the moment
    # this is currently O(n**2), but that optimization could cut in half the
    #  number of obstructing checks, but it would remain O(n**2)


################################################################################
##### TESTING ##################################################################
################################################################################
        



#seg = ((0,0),(1,0))
#obstructions = {(0.25,0),(0.75,0), (1.01, 0)}
#print chopSegWithObstructions((seg,obstructions))

#seg = ((0,8), (8,8))
#wall1 = ((0,-1), (-1,-1))
#print obstructSegViaSetOfSegs(eye, set([wall1]), seg)

eye = (2.5,3.5)
obSeg = ((1,5),(6,5))
seg1 = ((1,0),(1,4))
seg2 = ((2,3),(3,3))
seg3 = ((3,4),(4,4))
seg4 = ((3,3),(3,4))
s = set([seg1, seg2, seg3, seg4])

print obstructSegViaSetOfSegs(eye, s, obSeg)


#print chopSegWithObstructions(obstructSegViaSeg(eye, blockingSeg, [seg,set()]))
#print obstructSegWithObstructionsViaSeg(eye, blockingSeg, [seg,set()])
#print makeLine(eye, (1,1))

from Tkinter import *
root = Tk()
canvas = Canvas(root, width=500, height=500)
canvas.pack()

canvas.create_text(50*eye[0], 50*eye[1], text=".", fill="red", font="Times 18")

#canvas.create_line(50*eye[0], 50*eye[1], 50*seg1[0][0], 50*seg1[0][1], fill="red", width=2)
#canvas.create_line(50*eye[0], 50*eye[1], 50*seg1[1][0], 50*seg1[1][1], fill="red", width=2)
#canvas.create_line(50*eye[0], 50*eye[1], 50*seg2[0][0], 50*seg2[0][1], fill="red", width=2)
#canvas.create_line(50*eye[0], 50*eye[1], 50*seg2[1][0], 50*seg2[1][1], fill="red", width=2)
#canvas.create_line(50*eye[0], 50*eye[1], 50*seg3[0][0], 50*seg3[0][1], fill="red", width=2)
#canvas.create_line(50*eye[0], 50*eye[1], 50*seg3[1][0], 50*seg3[1][1], fill="red", width=2)
canvas.create_line(50*seg1[0][0], 50*seg1[0][1], 50*seg1[1][0], 50*seg1[1][1])
canvas.create_line(50*seg2[0][0], 50*seg2[0][1], 50*seg2[1][0], 50*seg2[1][1])
canvas.create_line(50*seg3[0][0], 50*seg3[0][1], 50*seg3[1][0], 50*seg3[1][1])
canvas.create_line(50*seg4[0][0], 50*seg4[0][1], 50*seg4[1][0], 50*seg4[1][1])
canvas.create_line(50*obSeg[0][0], 50*obSeg[0][1], 50*obSeg[1][0], 50*obSeg[1][1])

colors = ["blue", "green", "red", "yellow", "cyan"]
i = 0
for s in obstructSegViaSetOfSegs(eye, s, obSeg):
    canvas.create_line(50*s[0][0], 50*s[0][1], 50*s[1][0], 50*s[1][1], fill=colors[i], width=3)
    i += 1

root.mainloop()




