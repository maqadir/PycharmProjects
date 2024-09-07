data = [10, 20, 31.8, 1, 4, 5, 5, 34, 17, 76, 9]

sum_list = 0
avg_list = 0

for item in data:
    sum_list += item

avg_list = sum_list / len(data)

print('Sum = %d and mean =%.2f' % (sum_list, avg_list))
