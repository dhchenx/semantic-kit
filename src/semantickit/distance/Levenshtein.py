# coding: utf-8
import numpy as np

def edit_distance(a, b):
    m, n = len(a), len(b)
    dis_matrix = np.zeros((m+1, n+1), dtype=int)

    dis_matrix[0, :] = np.arange(n+1)
    dis_matrix[:, 0] = np.arange(m+1)

    for idx_a, ch_a in enumerate(a, 1):
        for idx_b, ch_b in enumerate(b, 1):
            cur_dis = None

            dis_left = dis_matrix[idx_a, idx_b-1]
            dis_upper = dis_matrix[idx_a-1, idx_b]
            dis_upper_left = dis_matrix[idx_a-1, idx_b-1]
            # print(ch_a,ch_b)
            if ch_a == ch_b:
                cur_dis = min(dis_left+1, dis_upper+1, dis_upper_left)
            else:
                cur_dis = min(dis_left+1, dis_upper+1, dis_upper_left + 1)

            dis_matrix[idx_a, idx_b] = cur_dis

    return dis_matrix[m, n]
