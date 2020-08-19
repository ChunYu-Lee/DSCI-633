"""
A regular polygon has n number of sides. Each side has length s.
*We want to get the sum of area and square of perimeter rounded to 4 decimal places.

area: 0.25 * n * (s**2)/math.tan(math.pi/n)
perimeter: n * s
"""
# import module math to get tan and pi function
import math

def polysum(n,s):

	# a --> the area of the polygon
    a = 0.25 * n * (s**2)/math.tan(math.pi/n)
    
    # b --> the perimeter of the polygon
    b = (n * s)**2

    # polysum = area of polygon + perimeter of the polygon
    return round(a+b,4)