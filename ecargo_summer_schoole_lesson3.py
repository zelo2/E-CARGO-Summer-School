import docplex.mp.model as cpx
import numpy as np
import time

'''
Start on: 2023/7/18
End on: 2023/7/19
@author: Zelo2
'''


def gra_cplex(stu_num, group_size):
    '''
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

    # Model initialization
    gra_model = cpx.Model(name="GRA lesson 3")

    t = {(i, j): gra_model.binary_var(name="x_{0}_{1}".format(i, j))
         for i in row for j in column}

    # Constraints
    for i in row:
        gra_model.add_constraint(gra_model.sum(t[i, j] for j in column) <= 1)  # one student one group

    for j in column:
        gra_model.add_constraint(gra_model.sum(t[i, j] for i in row) == group_size)  # group size constraint

    gra_model.add_constraint(gra_model.sum(t[i, j] * c[i][j] for i in row for j in column) == 0)  # conflict constraint

    # Define objective function
    objective = gra_model.sum(q[i][j] * t[i, j] * p[i][j] for i in row for j in column)
    gra_model.maximize(objective)

    # Problem Solving
    t0 = time.time()
    gra_model.solve()
    execution_time = time.time() - t0

    allocation_matrix = np.zeros([student_num, task_num])
    for i in row:
        for j in column:
            if t[i, j].solution_value == 1:
                allocation_matrix[i, j] = 1

    print("Cplex Allocation Results:")
    print(allocation_matrix)

    np.savetxt("q.txt", q, fmt='%.3f')
    np.savetxt("c.txt", c, fmt='%.0f')
    np.savetxt("p.txt", p, fmt='%.0f')
    np.savetxt("t.txt", allocation_matrix, fmt='%.0f')
    return gra_model.objective_value, execution_time, allocation_matrix


if __name__ == '__main__':
    result = []
    o_v, e_t, _ = gra_cplex(stu_num=15, group_size=3)
    print("Objecitve value:", o_v)
    print('Exection time(s):', e_t)
