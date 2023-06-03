def get_eq_triangle_coords(centroid=(0, 0), radius: float = 1):

    side_length = radius * 3**.5
    a = [centroid[0], centroid[1] + (3**.5 / 3) * side_length]  # top vertex
    b = [centroid[0] - (side_length / 2), centroid[1] - (3**.5 / 6) * side_length]  # bottom left vertex
    c = [centroid[0] + (side_length / 2), centroid[1] - (3**.5 / 6) * side_length]  # bottom right vertex

    return a, b, c

if __name__ == '__main__':

    import numpy as np
    coords = np.hstack([np.array(get_eq_triangle_coords(radius=1)), np.zeros([3, 1])])
    print(coords)
