import docplex.mp.model as cpx
import numpy as np
import time

'''
Start on: 2023/7/21
End on: 2023/7/21
@author: Zelo2
'''


def graccf_cplex(stu_num, group_size):
    '''
    Solving group role assignment with cooperation and conflict factors (GRACCF) problem via CPLEX solver
    :param stu_num: # of student
    :param group_size: # of group size
    :return: obejctive value, execution time, t-matrix
    '''
    student_num = stu_num
    task_num = int(stu_num / group_size)
    print("Student Num", student_num)
    print("Task Num:", task_num)
    print("Group Size:", group_size)

    row = range(stu_num)
    column = range(task_num)

    q = np.random.uniform(0, 1, size=[student_num, task_num])
    st_c = np.random.binomial(n=1, p=0.1, size=[student_num, task_num])  # student-task conflicts

    ss_c = np.random.uniform(-1, 1, size=[student_num * task_num, student_num * task_num])  # student-student conflicts
    sparse_mark = np.random.binomial(n=1, p=0.8, size=[student_num * task_num, student_num * task_num])
    # np.fill_diagonal(ss_c, 0)
    ss_c[sparse_mark == 1] = 0

    sp_ssc = []  # each element is a quintuple consisting of [id_1, id_1's role, id_2, id_2's role, cooperation effect]
    for i in range(student_num * task_num):  # i = stu_1 * task_num + task_j
        for j in range(student_num * task_num):
            i_1 = int(i // task_num)  # id of the first student
            j_1 = int(i_1 % task_num)  # id of the first student's role (role=group)

            i_2 = int(j // task_num)
            j_2 = int(i_2 % task_num)

            if i_1 != i_2 and ss_c[i, j] != 0:
                # print(i, j)
                sp_ssc.append([i_1, j_1, i_2, j_2, ss_c[i, j]])

    sp_ssc = np.array(sp_ssc)
    # print(np.where(ss_c != 0))
    n_c = sp_ssc.shape[0]  # significance number

    p = np.random.binomial(n=1, p=0.7, size=[student_num, task_num])
    dislike_mark = np.random.binomial(n=1, p=0.1, size=[student_num, task_num])
    p[dislike_mark == 1] = -1

    la = np.random.randint(1, 4, size=student_num)

    # Model initialization
    gmraccf_model = cpx.Model(name="GRA lesson 9/lab 4")

    t = {(i, j): gmraccf_model.binary_var(name="x_{0}_{1}".format(i, j))
         for i in row for j in column}
    t_plus = {i: gmraccf_model.binary_var(name="x_{0}_{1}".format(i, j))
              for i in range(n_c)}

    # Constraints
    for i in row:
        gmraccf_model.add_constraint(gmraccf_model.sum(t[i, j] for j in column) <= la[i])  # L^a constraints

    for j in column:
        gmraccf_model.add_constraint(gmraccf_model.sum(t[i, j] for i in row) == group_size)  # group size constraint

    gmraccf_model.add_constraint(
        gmraccf_model.sum(t[i, j] * st_c[i][j] for i in row for j in column) == 0)  # conflict constraint

    for i in range(n_c):
        gmraccf_model.add_constraint(
            (2 * t_plus[i]) <= t[int(sp_ssc[i, 0]), int(sp_ssc[i, 1])] + t[int(sp_ssc[i, 2]), int(sp_ssc[i, 3])])
        gmraccf_model.add_constraint(
            (t[int(sp_ssc[i, 0]), int(sp_ssc[i, 1])] + t[int(sp_ssc[i, 2]), int(sp_ssc[i, 3])]) <= t_plus[i] + 1)

    # Define objective function
    objective_general = gmraccf_model.sum(q[i][j] * t[i, j] * p[i][j] for i in row for j in column)
    # for i in range(n_c):
    #     print(sp_ssc[i, 0], sp_ssc[i, 1])
    #     print(q[int(sp_ssc[i, 0])][int(sp_ssc[i, 1])])
    #     print(type(sp_ssc[i, 0]), type(sp_ssc[i, 1]), type(sp_ssc[i, 2]), type(sp_ssc[i, 3]))

    objective_cooperation = gmraccf_model.sum(
        sp_ssc[i, 4] * q[int(sp_ssc[i, 0])][int(sp_ssc[i, 1])] * t_plus[i] for i in range(n_c))
    objective = objective_general + objective_cooperation
    gmraccf_model.maximize(objective)

    # Problem Solving
    t0 = time.time()
    gmraccf_model.solve()
    execution_time = time.time() - t0

    allocation_matrix = np.zeros([student_num, task_num])
    for i in row:
        for j in column:
            if t[i, j].solution_value == 1:
                allocation_matrix[i, j] = 1

    t_plus_result = np.zeros(n_c)
    for i in range(n_c):
        if t_plus[i].solution_value ==1:
            t_plus_result[i] = 1

    real_la = np.sum(allocation_matrix, axis=1)  # sum each row

    print("L^a matrix", la)
    print(real_la <= la)
    print("Cplex Allocation Results T:")
    print(allocation_matrix)
    print(np.sum(allocation_matrix, axis=0))
    print("Cplex T_plus Matrix：")
    print(t_plus_result)

    np.savetxt("q.txt", q, fmt='%.3f')
    np.savetxt("st_c.txt", st_c, fmt='%.0f')
    np.savetxt("ss_c.txt", ss_c, fmt='%.4f')
    np.savetxt("sparse_spssc.txt", sp_ssc, fmt='%.4f')
    np.savetxt("p.txt", p, fmt='%.0f')
    np.savetxt("t.txt", allocation_matrix, fmt='%.0f')
    np.savetxt("t_plus.txt", t_plus_result, fmt='%.0f')
    np.savetxt("L^a.txt", la, fmt='%.0f')
    return gmraccf_model.objective_value, execution_time, allocation_matrix


if __name__ == '__main__':
    result = []
    o_v, e_t, _ = graccf_cplex(stu_num=15, group_size=3)
    print("Objecitve value:", o_v)
    print('Exection time(s):', e_t)
