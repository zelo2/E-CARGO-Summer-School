import docplex.mp.model as cpx
import numpy as np
import time

'''
Start on: 2023/7/19
End on: 2023/7/19
@author: Zhuoxuan Huang
'''


def gmra_cplex(stu_num, group_size):
    '''
    Solving group multi-role assignment problem via CPLEX solver
    :param stu_num: # of student
    :param group_size: # of group size
    :return: obejctive value, execution time, t-matrix
    '''
    student_num = stu_num
    task_num = int(stu_num / group_size)

    row = range(stu_num)
    column = range(task_num)

    q = np.random.uniform(0, 1, size=[student_num, task_num])
    c = np.random.binomial(n=1, p=0.1, size=[student_num, task_num])

    p = np.random.binomial(n=1, p=0.7, size=[student_num, task_num])
    dislike_mark = np.random.binomial(n=1, p=0.1, size=[student_num, task_num])
    p[dislike_mark == 1] = -1

    la = np.random.uniform(1, 3, size=student_num)
    la = np.around(la)

    # Model initialization
    gmra_model = cpx.Model(name="GRA lesson 5")

    t = {(i, j): gmra_model.binary_var(name="x_{0}_{1}".format(i, j))
         for i in row for j in column}

    # Constraints
    for i in row:
        gmra_model.add_constraint(gmra_model.sum(t[i, j] for j in column) <= la[i])  # L^a constraints

    for j in column:
        gmra_model.add_constraint(gmra_model.sum(t[i, j] for i in row) == group_size)  # group size constraint

    gmra_model.add_constraint(
        gmra_model.sum(t[i, j] * c[i][j] for i in row for j in column) == 0)  # conflict constraint

    # Define objective function
    objective = gmra_model.sum(q[i][j] * t[i, j] * p[i][j] for i in row for j in column)
    gmra_model.maximize(objective)

    # Problem Solving
    t0 = time.time()
    gmra_model.solve()
    execution_time = time.time() - t0

    allocation_matrix = np.zeros([student_num, task_num])
    for i in row:
        for j in column:
            if t[i, j].solution_value == 1:
                allocation_matrix[i - 1, j - 1] = 1

    print("Cplex Allocation Results:")
    print(allocation_matrix)

    np.savetxt("q.txt", q, fmt='%.3f')
    np.savetxt("c.txt", c, fmt='%.0f')
    np.savetxt("p.txt", p, fmt='%.0f')
    np.savetxt("t.txt", allocation_matrix, fmt='%.0f')
    np.savetxt("L^a.txt", la, fmt='%.0f')
    return gmra_model.objective_value, execution_time, allocation_matrix


if __name__ == '__main__':
    # travel day = task_num / 2  旅行天数=任务数/2
    result = []
    o_v, e_t, _ = gmra_cplex(stu_num=15, group_size=3)
    print("Objecitve value:", o_v)
    print('Exection time(s):', e_t)
