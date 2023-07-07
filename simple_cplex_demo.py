import docplex.mp.model as cpx
import random
import numpy as np
import time


def cplex_demo(hotel_num, restaurant_num, attraction_num, task_num):
    '''
    :param hotel_num: 酒店数量
    :param restaurant_num: 餐厅数量
    :param attraction_num: 景点数量
    :param task_num: 任务数量
    :return: 休憩点分配矩阵
    '''
    rest_spot_num = hotel_num + restaurant_num

    set_rest_spot = range(1, rest_spot_num + 1)  # 1->poi_num
    set_attraction = range(1, attraction_num + 1)
    set_task = range(1, task_num + 1)  # 1-> task_num

    '''Initialization RSAM'''
    # 随机生成q矩阵 q_ij-the matching degree of rest spot #i on task #j
    # 前hotel_num行表示酒店和任务的匹配度，hotel_num + 1 - rest_spot_num 行表示餐厅和任务的匹配度
    q_rest_spot = {(i, j): random.uniform(0, 1) for i in set_rest_spot for j in set_task}
    rsam_model = cpx.Model(name="rest spot allocation module")

    # allocation matrix for rest spot allocation module (rsam)
    # size: [rest_spot_num, task_num], t_ij: whether rest spot #i is assigned to task #j
    rsam_t = {(i, j): rsam_model.binary_var(name="x_{0}_{1}".format(i, j))
              for i in set_rest_spot for j in set_task}

    # 随机生成冲突矩阵 (冲突概率=0.1) Conflict matrix 1-conflict exists 0-otherwise
    rsam_conflict = {(i, j): np.random.binomial(n=1, p=0.1, size=1)[0] for i in set_rest_spot for j in set_task}

    # Constraints
    # General constraints
    for i in set_rest_spot:
        rsam_model.add_constraint(rsam_model.sum(rsam_t[i, j] for j in set_task) <= 1)  # 每个休憩点只能被选择一次

    for j in range(1, task_num + 1, 2):
        rsam_model.add_constraint(rsam_model.sum(rsam_t[i, j] for i in range(1, hotel_num + 1)) == 1)  # 奇数任务必须包含餐厅

    for j in range(2, task_num + 1, 2):
        rsam_model.add_constraint(
            rsam_model.sum(rsam_t[i, j] for i in range(hotel_num + 1, rest_spot_num + 1)) == 1)  # 偶数任务必须包含酒店

    for j in set_task:
        rsam_model.add_constraint(rsam_model.sum(rsam_t[i, j] for i in set_rest_spot) == 1)  # 每个任务必须包含一个休憩点

    # Distance-aware constraints
    rsam_model.add_constraint(
        rsam_model.sum(
            rsam_conflict[i, j] * rsam_t[i, j] for i in set_rest_spot for j in set_task) == 0)  # 选择的休憩点和任务之间不存在冲突

    # Define objective function
    objective = rsam_model.sum(q_rest_spot[i, j] * rsam_t[i, j] for i in set_rest_spot for j in set_task)
    rsam_model.maximize(objective)

    # Problem Solving
    t0 = time.time()
    result_rsam = rsam_model.solve()
    print("Objective value (Profit) ", rsam_model.objective_value, "；Execution time/s ",
          time.time() - t0)

    '''Summarize the results'''
    allocation_matrix = np.zeros([rest_spot_num, task_num])
    for i in set_rest_spot:
        for j in set_task:
            if rsam_t[i, j].solution_value == 1:
                allocation_matrix[i - 1, j - 1] = 1

    print("Allocation Results:")
    print(allocation_matrix)

    # '''Initialization AAM'''
    # # q_ij-the matching degree of attraction #i on task #j
    # q_attraction = {(i, j): random.uniform(0, 1) for i in set_attraction for j in set_task}
    # aam_model = cpx.Model(name='attraction allocation module')
    #
    # # allocation matrix for attraction allocation module (aam)
    # t_aam = {(i, j): aam_model.binary_var(name="x_{0}_{1}".format(i, j))
    #           for i in set_rest_spot for j in set_task}

    # # Define Constraints
    # for j in set_J:
    #     opt_model.add_constraint(opt_model.sum(a[i, j] * x_vars[i, j] for i in set_I) <= b[j])
    #
    # for i in set_I:
    #     for j in set_J:
    #         opt_model.add_constraint(x_vars[i, j] >= l[i, j])
    #         opt_model.add_constraint(x_vars[i, j] <= u[i, j])


if __name__ == '__main__':
    # travel day = task_num / 2  旅行天数=任务数/2
    cplex_demo(hotel_num=50, restaurant_num=50, attraction_num=20, task_num=4)

    # a = np.array([0.6983,
    #               0.2120,
    #               0.6197,
    #               0.5772,
    #               0.2578,
    #               0.7547,
    #               0.7037,
    #               0.4283
    #               ])
    # b = np.array([0.7470, 0.7470, 0.7470, 0.7470,
    #               0.5438, 0.8772, 0.5438, 0.8772,
    #               0.6667, 1.0000, 0.6667, 1.0000,
    #               0.6667, 0.6667, 0.6667, 0.6667,
    #               0.6667, 0.6667, 0.6667, 0.6667,
    #               0.4942, 0.4942, 0.4942, 0.4942,
    #               0.3303, 0.6636, 0.3303, 0.6636,
    #               0.3429, 0.3429, 0.3429, 0.3429
    #               ])
    # c = np.array([0.9311, 1.0000, 0.9966, 0.9927,
    #               0.5051, 0.9897, 1.0000, 0.9976,
    #               0.8985, 1.0000, 1.0000, 1.0000,
    #               0.9008, 0.9312, 0.8923, 0.9980,
    #               1.0000, 1.0000, 0.9995, 0.9990,
    #               1.0000, 0.9994, 0.5719, 0.8938,
    #               0.9972, 1.0000, 1.0000, 0.9954,
    #               1.0000, 0.8755, 0.9998, 0.9998
    #               ])
    #
    # d = a.reshape(-1, 1) + b.reshape(8, 4) + c.reshape(8, 4)
    # d /= 3
    # print(d)
    # q_matrix = d
    # opt_model = cpx.Model()
    #
    # set_a = range(1, 9)
    # set_task = range(1, 5)
    # q = {(i, j): q_matrix[i - 1][j - 1] for i in set_a for j in set_task}
    # t = {(i, j): opt_model.binary_var(name="x_{0}_{1}".format(i, j))
    #           for i in set_a for j in set_task}
    #
    # # General constraints
    # for i in set_a:
    #     opt_model.add_constraint(opt_model.sum(t[i, j] for j in set_task) <= 1)
    #
    # for j in set_task:
    #     opt_model.add_constraint(opt_model.sum(t[i, j] for i in set_a) == 1)  # 每一列必须有一个1，即每个任务必须要分配景点
    #
    # # objective function
    # objective = opt_model.sum(q[i, j] * t[i, j] for i in set_a for j in set_task)
    # opt_model.maximize(objective)
    #
    # # Problem Solving
    # t0 = time.time()
    # mark = opt_model.solve()
    #
    # allocation_matrix = np.zeros([8, 4])
    # for i in set_a:
    #     for j in set_task:
    #         if t[i, j].solution_value == 1:
    #             allocation_matrix[i - 1, j - 1] = 1
    #
    # print("Allocation Results:")
    # print(allocation_matrix)


    # hotel_prefer = np.array([0.13043478, 0.13043478, 0.08695652, 0.13043478, 0.13043478,
    #                          0.2173913, 0.17391304])
    # resta_prefer = np.array([0.14285714, 0.0952381, 0.0952381, 0.14285714, 0.0952381,
    #                          0.23809524, 0.19047619])
    #
    # hotel = [0.878, 0.594, 0.542, 0.237, 0.211, 0.344, 0.770,
    #          0.116, 0.260, 0.875, 0.098, 0.144, 0.517, 0.380,
    #          0.797, 0.772, 0.691, 0.999, 0.245, 0.260, 0.135,
    #          0.469, 0.815, 0.195, 0.644, 0.651, 0.826, 0.104,
    #          0.538, 0.519, 0.655, 0.982, 0.070, 0.718, 0.041,
    #          0.055, 0.280, 0.697, 0.109, 0.402, 0.426, 0.353]
    #
    # restaurant = [0.609, 0.304, 0.983, 0.997, 0.879, 0.187, 0.633,
    #               0.137, 0.488, 0.814, 0.574, 0.571, 0.119, 0.215,
    #               0.189, 0.283, 0.724, 0.402, 0.319, 0.756, 0.116,
    #               0.705, 0.050, 0.491, 0.175, 0.138, 0.768, 0.977,
    #               0.890, 0.446, 0.145, 0.034, 0.154, 0.130, 0.564,
    #               0.605, 0.423, 0.618, 0.584, 0.991, 0.039, 0.776
    #               ]
    # hotel = np.array(hotel).reshape(6, -1)
    # restaurant = np.array(restaurant).reshape(6, -1)
    #
    # temp_h = hotel * hotel_prefer
    # temp_h = np.sum(temp_h, axis=1)
    # print(temp_h)
    #
    # temp_r = restaurant * resta_prefer
    # temp_r = np.sum(temp_r, axis=1)
    # print(temp_r)
