matrix =[[1,2,3],[4,5,6],[7,8,9]]
m, n = len(matrix), len(matrix[0])
left, right = 0, n
top, bottom = 0, m
res = []
# while left < right and top < bottom:
for i in range(left, right):
    res.append(matrix[0][i])
    top += 1
# for i in range(top, bottom):
#     res.append(matrix[i][right - 1])
#     right -= 1
#     if top < bottom:
#         for i in range(right, left - 1, -1):
#             res.append(matrix[bottom - 1][i])
#             bottom -= 1
#     if left < right:
#         for i in range(bottom, top - 1, -1):
#             res.append(matrix[i][left])
#             left += 1
print(res)