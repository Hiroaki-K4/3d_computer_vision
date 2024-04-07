def calculate_rows_of_dot_between_camera_mat_and_3d_position(P, pos_3d):
    X = pos_3d[0]
    Y = pos_3d[1]
    Z = pos_3d[2]
    p = P[0][0] * X + P[0][1] * Y + P[0][2] * Z + P[0][3]
    q = P[1][0] * X + P[1][1] * Y + P[1][2] * Z + P[1][3]
    r = P[2][0] * X + P[2][1] * Y + P[2][2] * Z + P[2][3]

    return p, q, r
