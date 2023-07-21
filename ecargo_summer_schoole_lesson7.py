import docplex.mp.model as cpx
import numpy as np
import time

'''
Start on: 2023/7/20
End on: 2023/7/21
@author: zelo2
'''


def gmracar_cplex(stu_num, group_size):
    '''
    Solving group role assignment with conflicting agents on roles (GRACAR) problem via CPLEX solver
    :param stu_num: # of student
    :param group_size: # of group size
    :return: obejctive value, execution time, t-matrix
    '''
    student_num = stu_num
    task_num = int(stu_num / group_size)

    row = range(stu_num)
    column = range(task_num)

    q = np.random.uniform(0, 1, size=[student_num, task_num])
    st_c = np.random.binomial(n=1, p=0.1, size=[student_num, task_num])  # student-task conflicts

    ss_c = np.random.binomial(n=1, p=0.5, size=[student_num, student_num])  # student-student conflicts
    np.fill_diagonal(ss_c, 0)

    p = np.random.binomial(n=1, p=0.7, size=[student_num, task_num])
    dislike_mark = np.random.binomial(n=1, p=0.1, size=[student_num, task_num])
    p[dislike_mark == 1] = -1

    la = np.random.randint(1, 4, size=student_num)

    # Model initialization
    gracar_model = cpx.Model(name="GRA lesson 7/lab 3")

    t = {(i, j): gracar_model.binary_var(name="x_{0}_{1}".format(i, j))
         for i in row for j in column}

    # Constraints
    for i in row:
        gracar_model.add_constraint(gracar_model.sum(t[i, j] for j in column) <= la[i])  # L^a constraints

    for j in column:
        gracar_model.add_constraint(gracar_model.sum(t[i, j] for i in row) == group_size)  # group size constraint

    gracar_model.add_constraint(
        gracar_model.sum(t[i, j] * st_c[i][j] for i in row for j in column) == 0)  # conflict constraint

    for i in row:  # student-student conflict constraint
        for j in column:
            for x in row:
                if i != x:
                    gracar_model.add_constraint(gracar_model.sum((t[i, j] + t[x, j]) * ss_c[i][x]) <= 1)

    # Define objective function
    objective = gracar_model.sum(q[i][j] * t[i, j] * p[i][j] for i in row for j in column)
    gracar_model.maximize(objective)

    # Problem Solving
    t0 = time.time()
    gracar_model.solve()
    execution_time = time.time() - t0

    allocation_matrix = np.zeros([student_num, task_num])
    for i in row:
        for j in column:
            if t[i, j].solution_value == 1:
                allocation_matrix[i, j] = 1

    real_la = np.sum(allocation_matrix, axis=1)  # sum each row

    print("L^a matrix", la)
    print(real_la <= la)
    print("Cplex Allocation Results:")
    print(allocation_matrix)

    np.savetxt("q.txt", q, fmt='%.3f')
    np.savetxt("st_c.txt", st_c, fmt='%.0f')
    np.savetxt("ss_c.txt", ss_c, fmt='%.0f')
    np.savetxt("p.txt", p, fmt='%.0f')
    np.savetxt("t.txt", allocation_matrix, fmt='%.0f')
    np.savetxt("L^a.txt", la, fmt='%.0f')
    return gracar_model.objective_value, execution_time, allocation_matrix


if __name__ == '__main__':
    result = []
    o_v, e_t, _ = gmracar_cplex(stu_num=15, group_size=3)
    print("Objecitve value:", o_v)
    print('Exection time(s):', e_t)
