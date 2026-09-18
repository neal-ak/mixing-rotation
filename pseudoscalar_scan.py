# for a mass we want a number of coupling points x:
# for point x1, find nearest point x2 with solutions
# check if x1 point is within some abs(x1/x2 - 1)<threshold
# if it is, use x2 solutions as unknowns to guess
# else use the approximations, falling back to defaults if approximations give errors
