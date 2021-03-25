import numpy as np


def create_matrix(L, m, u, n):
    """
    Создание матрицы по заданным параметрам

    Args:
        L (int): интенсивность поступления λ
        m (int): кол-во каналов обслуживания m
        u (int): интенсивность обслуживания μ
        n (int): максимальный размер очереди
    """
    p_matrix = np.zeros((m + n + 1, m + n + 1))
    for i in range(m + n):
        p_matrix[i, i+1] = L
        if i < m:
            p_matrix[i + 1, i] = (u * (i + 1))
        else:
            p_matrix[i + 1, i] = u*m
    return p_matrix


def task_a(matrix):
    """
    Установившиеся вероятности
    """
    diag_res = []
    for i in range(matrix.shape[0]):
        diag_res.append(matrix[i, :].sum())

    D = np.diag(diag_res)
    M = matrix.transpose() - D
    M_ = np.copy(M)
    M_[-1, :] = 1

    b_vector = np.zeros(M_.shape[0])
    b_vector[-1] = 1
    X = np.linalg.inv(M_).dot(b_vector)
    return X


def task_b(vector):
    """
    Вероятность отказа в обслуживании
    """
    return vector[-1]


def task_c(vector, L):
    """
    Относительная и абсолютная интенсивность обслуживания
    """
    relative = 1 - vector[-1]
    return relative, relative * L


def task_d(vector, m, n):
    """
    Средняя длина очереди
    """
    s = 0
    for i in range(1, n+1):
        s += i * vector[m+i]
    return s


def task_e(vector, m, u, n):
    """
    Среднее время в очереди
    """
    s = 0
    for i in range(n):
        s += ((i+1)/(m*u)*vector[m+i])
    return s


def task_f(vector, m, n):
    """
    Среднее число занятых каналов
    """
    s = 0
    for i in range(1, m+n+1):
        if i <= m:
            s += i * vector[i]
        else:
            s += m * vector[i]
    return s


def task_g(vector, m):
    """
    Вероятность не ждать в очереди
    """
    return sum(vector[:m])


def task_h(matrix):
    """
    Среднее время простоя системы массового обслуживания
    """
    return 1 / np.sum(matrix, -1)


L = 10
m = 7
u = 1
n = 15
matrix = create_matrix(L, m, u, n)

# a
vector = task_a(matrix)
print(f"Составьте граф марковского процесса, запишите систему уравнений Колмогорова и \
найдите установившиеся вероятности состояний:\n--> {vector}")

# b
answer_b = task_b(vector)
print(f"Найдите вероятность отказа в обслуживании:\n--> {answer_b}")

# c
relative, absolute = task_c(vector, L)
print("Найдите относительную и абсолютную интенсивность обслуживания:")
print(f"Относительная: {relative}\nАбсолютная: {absolute}")

# d
answer_d = task_d(vector, m, n)
print(f"Найдите среднюю длину в очереди:\n--> {answer_d}")

# e
answer_e = task_e(vector, m, u, n)
print(f"Найдите среднее время в очереди:\n--> {answer_e}")

# f
answer_f = task_f(vector, m, n)
print(f"Найдите среднее число занятых каналов:\n--> {answer_f}")

# g
answer_g = task_g(vector, m)
print(
    f"Найдите вероятность того, что поступающая заявка не будет ждать в очереди:\n--> {answer_g}")

# h
answer_h = task_h(matrix)
print(
    f"Найти среднее время простоя системы массового обслуживания:\n--> {answer_h[0]}")
