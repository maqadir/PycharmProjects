data = [10, 20, 31.8, 1, 4, 5, 5, 34, 17, 76, 9]

list_test = [1, 2, 3, 3, 4, 5, 5, 6, 7, 7]

sum_list = 0
avg_list = 0

for item in data:
    sum_list += item

avg_list = sum_list / len(data)

print('Sum = %d and mean =%.2f' % (sum_list, avg_list))


def unique_list(list_test):
    new_list = list(set(list_test))
    print(sorted(new_list))

list_test = [1, 2, 3, 3, 4, 5, 5, 6, 7, 7]
unique_list(list_test)
