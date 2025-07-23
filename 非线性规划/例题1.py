import numpy as np
from scipy.optimize import minimize

# 目标函数
def fun(vars):
    x, y = vars
    return (x - 1)**2 + (y - 2)**4

# 约束条件
cons = ({
    'type': 'ineq',
    'fun': lambda vars: vars[0] + vars[1] - 3  # x + y >= 3
})

# 变量范围
bounds = [(0, None), (0, None)]  # x >= 0, y >= 0

# 初始猜测
x0 = [0.5, 2.5]

result = minimize(fun, x0, constraints=cons, bounds=bounds, method='SLSQP')

print("最优解：", result.x)
print("最优目标值：", result.fun)