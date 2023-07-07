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


if __name__ == '__main__':
    # travel day = task_num / 2  旅行天数=任务数/2
    cplex_demo(hotel_num=50, restaurant_num=50, attraction_num=20, task_num=4)
