# Примеры задач и решений по каждому указанному алгоритму сложности:

# O(1) — Константное время
# Задача 1: Проверить, четное ли число.
# Решение:
def is_even(number):
    return number % 2 == 0


print(is_even(4))  # True
print(is_even(5))  # False

# Задача 2: Получить элемент массива по индексу.
# Решение:
arr = [10, 20, 30, 40, 50]
print(arr[2])  # 30

# Задача 3: Поменять значение двух переменных местами.
# Решение:
i = 5
j = 3
i, j = j, i


# Задача 4: Определить наибольшее из трех чисел.
# Решение:
def big(a, b, c):
    if a > b:
        return a if a > c else c
    else:
        return b if b > c else c


# Задача 5: Проверка, кратно ли число заданному делителю.
# Решение:
def is_divisible(number, divisor):
    return number % divisor == 0


print(is_divisible(10, 5))  # True
print(is_divisible(10, 3))  # False


# O(n) — Линейное время
# Задача 1: Найти максимальное число в массиве.
# Решение:
def find_max(arr):
    max_val = arr[0]
    for num in arr:
        if num > max_val:
            max_val = num
    return max_val


print(find_max([3, 1, 4, 1, 5, 9]))  # 9


# Задача 2: Посчитать сумму всех элементов в массиве.
# Решение:
def sum_array(arr):
    total = 0
    for num in arr:
        total += num
    return total


print(sum_array([1, 2, 3, 4, 5]))  # 15


# Задача 3: Сумма четных чисел.
# Решение:
def sum_che(m):
    mysum = 0
    for i in m:
        if i % 2 == 0:
            mysum += i
    return mysum


# Задача 4: Проверка строки на палиндром.
# Решение:
def is_palindrome(s):
    n = len(s)
    for i in range(n // 2):
        if s[i] != s[n - i - 1]:
            return False
    return True


print(is_palindrome("radar"))  # True
print(is_palindrome("hello"))  # False


def strfunc(x):
    return x == x[::-1]


# Задача 5: Фибоначчи.
# Решение:
def fib(n):
    f0, f1 = 0, 1
    for _ in range(n - 1):
        f0, f1 = f1, f0 + f1
        print(f1, end=' ')
    return f1


print(fib(7))

# Задача 6: Для проверки анаграмм можно избежать сортировки,
# используя частотный подсчет символов. Это улучшает алгоритм до линейной сложности:
# Решение:
from collections import Counter


def anagramm_optimized(a, b):
    return Counter(a) == Counter(b)


# Задача 7: Подсчет гласных букв.
# Решение:
def strl(x):
    g = "AEIOUaeiou"
    return len([i for i in x if i in g])


# Задача 8: Среднее арифметическое списка.
# Решение:
def lis(x):
    return sum(x) // len(x)


# Задача 9: Проверить, является ли число простым.
# Решение:
def easy(x):
    for i in range(2, x):
        if x % i == 0:
            return False
    return True


# Задача 10: Удаление переданного символа из строки.
# Решение:
def del_str(a: str, b: str):
    return a.replace(b, "")


# Задача 11: Подсчет вхождений подстроки в строку.
# Решение:
def count_str(a, b):
    return a.count(b)


# Задача 12: Число армстронга.
# Решение:
def arm(x):
    sum = 0
    size = len(str(x))
    for i in list(str(x)):
        sum += int(i) ** size
    return x == sum


print(arm(371))  # 371=3^3+7^3+1^3


# Задача 13: Инвертирование числа:
# Решение:
def int_slice(x):
    return int(str(x)[::-1])


# Задача 14: Проверить, есть ли в массиве повторяющиеся элементы.
# Решение:
def has_duplicates(arr):
    seen = set()
    for num in arr:
        if num in seen:
            return True
        seen.add(num)
    return False


print(has_duplicates([1, 2, 3, 4, 5]))  # False
print(has_duplicates([1, 2, 3, 4, 1]))  # True


# O(log n) — Логарифмическое время
# Задача 1: Найти элемент в отсортированном массиве с помощью бинарного поиска.
# Решение:
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


print(binary_search([1, 3, 5, 7, 9], 5))  # 2


# O(n^2) — Квадратичное время
# Задача 1: Отсортировать массив с помощью пузырьковой сортировки.
# Решение:
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


print(bubble_sort([5, 2, 9, 1, 5, 6]))  # [1, 2, 5, 5, 6, 9]


# Задача 2: Найти все пары элементов в массиве.
# Решение:
def find_pairs(arr):
    pairs = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            pairs.append((arr[i], arr[j]))
    return pairs


print(find_pairs([1, 2, 3]))  # [(1, 2), (1, 3), (2, 3)]


# Задача 3: Найти три числа в массиве, сумма которых равна заданному числу.
# Решение:
def three_sum(arr, target):
    n = len(arr)
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if arr[i] + arr[j] + arr[k] == target:
                    result.append((arr[i], arr[j], arr[k]))
    return result


print(three_sum([1, 2, 3, 4, 5, 6], 10))  # [(1, 3, 6), (2, 3, 5), (1, 4, 5)]


# O(n^3) — Кубическое время
# Задача 1: Реализовать умножение двух матриц.
# Решение:
def matrix_multiply(A, B):
    n, m, p = len(A), len(B), len(B[0])
    result = [[0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                result[i][j] += A[i][k] * B[k][j]
    return result


A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
print(matrix_multiply(A, B))  # [[19, 22], [43, 50]]


# O(n log n) — Линейно-логарифмическое время
# Задача 1: Отсортировать массив с помощью сортировки слиянием.
# Решение:
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


print(merge_sort([5, 2, 9, 1, 5, 6]))  # [1, 2, 5, 5, 6, 9]


# Задача 2: Анаграммы.
# Решение:
def anagramm(a, b):
    return sorted(a) == sorted(b)


# O(2^n) — Экспоненциальное время
# Задача 1: Найти все подмножества множества.
# Решение:
def find_subsets(nums):
    result = []

    def backtrack(index, current):
        if index == len(nums):
            result.append(current[:])
            return
        current.append(nums[index])
        backtrack(index + 1, current)
        current.pop()
        backtrack(index + 1, current)

    backtrack(0, [])
    return result


print(find_subsets([1, 2, 3]))  # [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]


# Задача 2: Фибоначчи рекурсия.
# Решение:
def fib1(n):
    if n in (1, 2):
        return 1
    return fib1(n - 1) + fib1(n - 2)


# Задача 3: Найти длину самой длинной общей подпоследовательности двух строк.
# Решение:
def lcs_recursive(a, b):
    if not a or not b:
        return 0
    if a[-1] == b[-1]:
        return 1 + lcs_recursive(a[:-1], b[:-1])
    return max(lcs_recursive(a[:-1], b), lcs_recursive(a, b[:-1]))


print(lcs_recursive("abcde", "ace"))  # 3 ("ace")


# O(n!) — Факториальное время
# Задача 1: Найти все перестановки массива.
# Решение:
def permute(nums):
    result = []

    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result


print(permute([1, 2, 3]))  # [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 2, 1], [3, 1, 2]]


# Задача 2: Генерация всех возможных скобочных последовательностей.
# Решение:
def generate_parentheses(n):
    result = []

    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return
        if open_count < n:
            backtrack(current + "(", open_count + 1, close_count)
        if close_count < open_count:
            backtrack(current + ")", open_count, close_count + 1)

    backtrack("", 0, 0)
    return result


print(generate_parentheses(3))  # ["((()))", "(()())", "(())()", "()(())", "()()()"]

# O(sqrt(n)) — Квадратный корень
# Задача 1: Проверить, является ли число простым.
# Решение:
import math


def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


print(is_prime(30))  # False
