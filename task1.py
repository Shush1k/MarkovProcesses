import numpy as np
from functools import lru_cache


def task1(matrix, k):
    """
    Вероятность перехода из состояния i в j за k шагов

    Args:
        matrix (numpy.ndarray): матрица переходов
        k (int): кол-во шагов
    """
    return np.linalg.matrix_power(matrix, k)


def task2(matrix, a_0, k):
    """
    Вероятность состояния за k шагов

    Args:
        matrix (numpy.ndarray): матрица переходов
        a_0 (numpy.ndarray): вероятности состояние в начальный момент времениъ
        k (int): кол-во шагов
    """
    return a_0.dot(np.linalg.matrix_power(matrix, k))


def task3(matrix, k):
    """
    Вероятность первого перехода за k шагов

    Args:
        matrix (numpy.ndarray): матрица переходов
        k (int): кол-во шагов
    """
    matrix_p = np.copy(matrix)
    for _ in range(1, k):
        matrix_p = skip_j_state(matrix, matrix_p)
    return matrix_p


def skip_j_state(matrix, matrix_pr):
    """
    Пропуск j состояния

    Args:
        matrix (numpy.ndarray): матрица переходов
        matrix_pr (numpy.ndarray): матрица переходов предыдущая
    """
    len_p = len(matrix)
    new_matrix = np.zeros((len_p, len_p))
    for i in range(len_p):
        for j in range(len_p):
            s = 0
            for m in range(len_p):
                if m != j:
                    s += matrix[i, m] * matrix_pr[m, j]
            new_matrix[i, j] = s
    matrix_pr = new_matrix
    return matrix_pr


def task4(matrix, k):
    """
    Вероятность перехода не позднее чем за k шагов

    Args:
        matrix (numpy.ndarray): матрица переходов
        k (int): кол-во шагов
    """
    matrix_pr, result = np.copy(matrix), np.copy(matrix)
    for _ in range(1, k):
        matrix_pr = skip_j_state(matrix, matrix_pr)
        result += matrix_pr
    return result


def task5(matrix):
    """
    Среднее количество шагов для перехода из состояния i в j

    Args:
        matrix (numpy.ndarray): матрица переходов
    """
    matrix_pr, result = np.copy(matrix), np.copy(matrix)
    for g in range(1, 700):
        matrix_pr = skip_j_state(matrix, matrix_pr)
        result += g * matrix_pr
    return result


def task6(matrix, k):
    """
    Вероятность первого возвращения на k-ом шаге

    Args:
        matrix (numpy.ndarray): матрица переходов
        k (int): кол-во шагов
    """
    matrix_pr = np.copy(matrix)
    @lru_cache(maxsize=None)
    def f_jj(k):
        return np.linalg.matrix_power(matrix_pr, k) - sum([f_jj(m) * np.linalg.matrix_power(matrix_pr, k - m) for m in range(1, k)])

    return np.diagonal(f_jj(k))


def task7(matrix, k):
    """
    Вероятность возвращения не позднее чем за k шагов

    Args:
        matrix (numpy.ndarray): матрица переходов
        k (int): кол-во шагов
    """
    matrix_pr = np.copy(matrix)
    result = []

    @lru_cache(maxsize=None)
    def f_jj(k):
        res = np.linalg.matrix_power(
            matrix_pr, k) - sum([f_jj(m) * np.linalg.matrix_power(matrix_pr, k - m) for m in range(1, k)])
        result.append(np.diagonal(res))
        return res

    f_jj(k)
    return sum(result)


def task8(matrix):
    """
    Среднее время возвращения

    Args:
        matrix (numpy.ndarray): матрица переходов
    """
    matrix_pr, result = np.copy(matrix), []

    @lru_cache(maxsize=None)
    def f_jj(k=500):
        res = np.linalg.matrix_power(
            matrix_pr, k) - sum([f_jj(m) * np.linalg.matrix_power(matrix_pr, k - m) for m in range(1, k)])
        result.append(k * np.diagonal(res))
        return res
    f_jj()
    return sum(result)


def task9(matrix):
    """
    Установившиеся вероятности

    Args:
        matrix (numpy.ndarray): матрица переходов
    """
    matrix_ = np.copy(matrix).transpose()
    np.fill_diagonal(matrix_, np.diagonal(matrix_) - 1)
    matrix_[-1, :] = 1

    b_vector = np.zeros(len(matrix))
    b_vector[-1] = 1
    X = np.linalg.inv(matrix_).dot(b_vector)
    return X


matrix = np.array([
    [0.05, 0.06, 0, 0, 0.3, 0.2, 0.39, 0, 0, 0, 0, 0, 0, 0],
    [0, 0.55, 0, 0, 0.45, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0.49, 0.14, 0.37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0.35, 0.39, 0.24, 0, 0.02, 0, 0, 0, 0, 0, 0, 0],
    [0.01, 0, 0.12, 0.17, 0.04, 0.09, 0, 0.14, 0.19, 0.08, 0.16, 0, 0, 0],
    [0, 0.12, 0, 0, 0, 0.35, 0, 0.39, 0.14, 0, 0, 0, 0, 0],
    [0, 0.52, 0.42, 0, 0, 0, 0.06, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0.26, 0, 0.08, 0.66, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0.23, 0, 0.28, 0.22, 0, 0, 0.27, 0, 0],
    [0, 0, 0, 0, 0.32, 0, 0.62, 0, 0, 0.06, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.74, 0, 0.16, 0, 0, 0, 0.1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0.19, 0, 0.24, 0.19, 0.09, 0, 0.29],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.12, 0.88],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.33, 0.6, 0.07]
])

print(f"Матрица переходов:\n{matrix}")

# 1
k, i, j = 10, 7, 4
answer1 = task1(matrix, k)
print(
    f"Вероятность того, что за k={k} шагов система перейдет из состояния {i} в состояние {j} \n--> {answer1[i-1][j-1]}")

# 2
k = 5
a_0 = np.array([0.09, 0.14, 0.09, 0.11, 0.03, 0.03, 0.15,
                0.05, 0.01, 0.03, 0.01, 0.14, 0.1, 0.02])
answer2 = task2(matrix, a_0, k)
print(f"Вероятности состояний системы спустя k={k} шагов, если в начальный "
      f"момент вероятность состояний были следующими \nA={a_0}\n\nОтвет: {answer2}")

# 3
k, i, j = 7, 10, 11
answer3 = task3(matrix, k)
print(
    f"Вероятность первого перехода за k={k} шагов из состояния {i} в состояние {j} \n--> {answer3[i-1][j-1]}")

# 4
i, j, k = 7, 9, 10
answer4 = task4(matrix, k)
print(
    f"Вероятность перехода из состояния {i} в состояние {j} не позднее чем за k={k} шагов \n--> {answer4[i-1][j-1]}")

# 5
i, j = 8, 10
answer5 = task5(matrix)
print(
    f"Среднее количество шагов для перехода из состояния {i} в состояние {j} \n--> {answer5[i-1][j-1]}")

# 6
i, k = 7, 6
answer6 = task6(matrix, k)
print(
    f"Вероятность первого возвращения в состояние {i} за k={k} шагов\n--> {answer6[i-1]}")

# 7
i, k = 2, 10
answer7 = task7(matrix, k)
print(
    f"Вероятность возвращения в состояние {i} не позднее чем за k={k} шагов\n--> {answer7[i-1]}")

# 8
i = 7
answer8 = task8(matrix)
print(f"Среднее время возвращения в состояние {i} \n--> {answer8[i-1]}")

# 9
answer9 = task9(matrix)
print(f"Установившиеся вероятности:\n--> {answer9}")
