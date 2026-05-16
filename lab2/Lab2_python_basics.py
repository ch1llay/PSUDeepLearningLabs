from random import randint

def generate_list(a, b, size):
    return [randint(a, b) for _ in range(size)]

def sum_of_even(list):
    even_sum = 0
    for num in list:
        if num % 2 == 0:
            even_sum += num
    return even_sum

def main() -> None:
    list = generate_list(1, 100, 100)
    even_sum = sum_of_even(list)

    print(f"Список чисел: {list}")
    print(f"Сумма четных чисел: {even_sum}")

    pass

if __name__ == '__main__':
    main()