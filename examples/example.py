from roughsurf2stl.roughsurface import gaussian_correlated_surface

# Shape properties
nx = 600
ny = 600
dx = 1.0
corr_len_x = 20.0
corr_len_y = 20.0
height_rms = 2.0

seed = 200
debug = True

gaussian_correlated_surface(nx, ny, dx, corr_len_x, corr_len_y, height_rms, seed, debug)
