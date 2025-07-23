# # # # # # # # # import numpy as np
# # # # # # # # # from scipy.optimize import linprog
# # # # # # # # #
# # # # # # # # import numpy as np
# # # # # # # # from scipy.optimize import minimize, linprog
# # # # # # # #
# # # # # # # # # 料场和工地坐标
# # # # # # # # depots = np.array([[5, 1], [2, 7]])
# # # # # # # # sites = np.array([[1.25, 1.25], [8.75, 0.75], [0.5, 4.75], [5.75, 5], [3, 6.5], [7.25, 7.25]])
# # # # # # # # demands = np.array([3, 5, 4, 7, 6, 11])
# # # # # # # # supplies = np.array([20, 20])
# # # # # # # #
# # # # # # # # # 距离矩阵
# # # # # # # # dist = np.linalg.norm(depots[:, None, :] - sites[None, :, :], axis=2)  # shape: (2,6)
# # # # # # # # cost = dist.flatten()  # 12维
# # # # # # # #
# # # # # # # # # 变量顺序：A->6工地, B->6工地
# # # # # # # # A = []
# # # # # # # # # 工地需求约束
# # # # # # # # for j in range(6):
# # # # # # # #     row = [0]*12
# # # # # # # #     row[j] = 1      # A->j
# # # # # # # #     row[6+j] = 1    # B->j
# # # # # # # #     A.append(row)
# # # # # # # # b = demands.tolist()
# # # # # # # #
# # # # # # # # # 料场供应约束
# # # # # # # # for i in range(2):
# # # # # # # #     row = [0]*12
# # # # # # # #     for j in range(6):
# # # # # # # #         row[i*6 + j] = 1
# # # # # # # #     A.append(row)
# # # # # # # # b += supplies.tolist()
# # # # # # # #
# # # # # # # # bounds = [(0, None)]*12
# # # # # # # #
# # # # # # # # res = linprog(cost, A_eq=A[:6], b_eq=b[:6], A_ub=A[6:], b_ub=b[6:], bounds=bounds, method='highs')
# # # # # # # # print("最优分配（每个变量对应A1~A6,B1~B6）：", res.x)
# # # # # # # # print("最小吨千米数：", res.fun)
# # # # # # # #
# # # # # # # # # sites = np.array([[1.25, 1.25], [8.75, 0.75], [0.5, 4.75], [5.75, 5], [3, 6.5], [7.25, 7.25]])
# # # # # # # # # demands = np.array([3, 5, 4, 7, 6, 11])
# # # # # # # # # supplies = np.array([20, 20])
# # # # # # # # #
# # # # # # # # # def total_tkm(depot_coords):
# # # # # # # # #     depots = depot_coords.reshape(2,2)
# # # # # # # # #     dist = np.linalg.norm(depots[:, None, :] - sites[None, :, :], axis=2)
# # # # # # # # #     cost = dist.flatten()
# # # # # # # # #     A = []
# # # # # # # # #     for j in range(6):
# # # # # # # # #         row = [0]*12
# # # # # # # # #         row[j] = 1
# # # # # # # # #         row[6+j] = 1
# # # # # # # # #         A.append(row)
# # # # # # # # #     b = demands.tolist()
# # # # # # # # #     for i in range(2):
# # # # # # # # #         row = [0]*12
# # # # # # # # #         for j in range(6):
# # # # # # # # #             row[i*6 + j] = 1
# # # # # # # # #         A.append(row)
# # # # # # # # #     b += supplies.tolist()
# # # # # # # # #     bounds = [(0, None)]*12
# # # # # # # # #     res = linprog(cost, A_eq=A[:6], b_eq=b[:6], A_ub=A[6:], b_ub=b[6:], bounds=bounds, method='highs')
# # # # # # # # #     return res.fun
# # # # # # # # #
# # # # # # # # # # 初始猜测：原料场位置
# # # # # # # # # x0 = np.array([5,1,2,7])
# # # # # # # # # result = minimize(total_tkm, x0, method='Nelder-Mead')
# # # # # # # # # print("新料场坐标：", result.x.reshape(2,2))
# # # # # # # # # print("最小吨千米数：", result.fun)
# # # # # # # #
# # # # # # # # # import numpy as np
# # # # # # # # # from scipy.optimize import minimize, linprog
# # # # # # # # #
# # # # # # # # # sites = np.array([[1.25, 1.25], [8.75, 0.75], [0.5, 4.75], [5.75, 5], [3, 6.5], [7.25, 7.25]])
# # # # # # # # # demands = np.array([3, 5, 4, 7, 6, 11])
# # # # # # # # # supplies = np.array([20, 20])
# # # # # # # # #
# # # # # # # # # def total_tkm(depot_coords):
# # # # # # # # #     depots = depot_coords.reshape(2,2)
# # # # # # # # #     dist = np.linalg.norm(depots[:, None, :] - sites[None, :, :], axis=2)
# # # # # # # # #     cost = dist.flatten()
# # # # # # # # #     A = []
# # # # # # # # #     for j in range(6):
# # # # # # # # #         row = [0]*12
# # # # # # # # #         row[j] = 1
# # # # # # # # #         row[6+j] = 1
# # # # # # # # #         A.append(row)
# # # # # # # # #     b = demands.tolist()
# # # # # # # # #     for i in range(2):
# # # # # # # # #         row = [0]*12
# # # # # # # # #         for j in range(6):
# # # # # # # # #             row[i*6 + j] = 1
# # # # # # # # #         A.append(row)
# # # # # # # # #     b += supplies.tolist()
# # # # # # # # #     bounds = [(0, None)]*12
# # # # # # # # #     res = linprog(cost, A_eq=A[:6], b_eq=b[:6], A_ub=A[6:], b_ub=b[6:], bounds=bounds, method='highs')
# # # # # # # # #     return res.fun
# # # # # # # # #
# # # # # # # # # # 初始猜测：原料场位置
# # # # # # # # # x0 = np.array([5,1,2,7])
# # # # # # # # # result = minimize(total_tkm, x0, method='Nelder-Mead')
# # # # # # # # # print("新料场坐标：", result.x.reshape(2,2))
# # # # # # # # # print("最小吨千米数：", result.fun)
# # # # # # #
# # # # # # # # python
# # # # # # # import numpy as np
# # # # # # # from scipy.optimize import minimize, linprog
# # # # # # #
# # # # # # # sites = np.array([
# # # # # # #     [1.25, 1.25],
# # # # # # #     [8.75, 0.75],
# # # # # # #     [0.5, 4.75],
# # # # # # #     [5.75, 5],
# # # # # # #     [3, 6.5],
# # # # # # #     [7.25, 7.25]
# # # # # # # ])
# # # # # # # demands = np.array([3, 5, 4, 7, 6, 11])
# # # # # # # supplies = np.array([20, 20])
# # # # # # #
# # # # # # # def total_tkm(depot_coords):
# # # # # # #     depots = depot_coords.reshape(2,2)
# # # # # # #     dist = np.linalg.norm(depots[:, None, :] - sites[None, :, :], axis=2)
# # # # # # #     cost = dist.flatten()
# # # # # # #     A = []
# # # # # # #     for j in range(6):
# # # # # # #         row = [0]*12
# # # # # # #         row[j] = 1
# # # # # # #         row[6+j] = 1
# # # # # # #         A.append(row)
# # # # # # #     b = demands.tolist()
# # # # # # #     for i in range(2):
# # # # # # #         row = [0]*12
# # # # # # #         for j in range(6):
# # # # # # #             row[i*6 + j] = 1
# # # # # # #         A.append(row)
# # # # # # #     b += supplies.tolist()
# # # # # # #     bounds = [(0, None)]*12
# # # # # # #     res = linprog(
# # # # # # #         cost,
# # # # # # #         A_eq=A[:6], b_eq=b[:6],
# # # # # # #         A_ub=A[6:], b_ub=b[6:],
# # # # # # #         bounds=bounds,
# # # # # # #         method='highs',
# # # # # # #         options={'presolve': True}
# # # # # # #
# # # # # # #     )
# # # # # # #     return np.round(res.fun, 8)  # 保留8位小数
# # # # # # #
# # # # # # # x0 = np.array([5,1,2,7])
# # # # # # # result = minimize(total_tkm, x0, method='Nelder-Mead', options={'xatol':1e-8, 'fatol':1e-8})
# # # # # # # print("新料场坐标：", np.round(result.x.reshape(2,2), 8))
# # # # # # # print("最小吨千米数：", np.round(result.fun, 8))
# # # # # #
# # # # # # # python
# # # # # # import numpy as np
# # # # # # from scipy.optimize import minimize, linprog
# # # # # #
# # # # # # sites = np.array([
# # # # # #     [1.25, 1.25],
# # # # # #     [8.75, 0.75],
# # # # # #     [0.5, 4.75],
# # # # # #     [5.75, 5],
# # # # # #     [3, 6.5],
# # # # # #     [7.25, 7.25]
# # # # # # ])
# # # # # # demands = np.array([3, 5, 4, 7, 6, 11])
# # # # # # supplies = np.array([20, 20])
# # # # # #
# # # # # # # 返回：吨千米数、分配方案
# # # # # # def total_tkm(depot_coords):
# # # # # #     depots = depot_coords.reshape(2,2)
# # # # # #     dist = np.linalg.norm(depots[:, None, :] - sites[None, :, :], axis=2)
# # # # # #     cost = dist.flatten()
# # # # # #     A = []
# # # # # #     for j in range(6):
# # # # # #         row = [0]*12
# # # # # #         row[j] = 1
# # # # # #         row[6+j] = 1
# # # # # #         A.append(row)
# # # # # #     b = demands.tolist()
# # # # # #     for i in range(2):
# # # # # #         row = [0]*12
# # # # # #         for j in range(6):
# # # # # #             row[i*6 + j] = 1
# # # # # #         A.append(row)
# # # # # #     b += supplies.tolist()
# # # # # #     bounds = [(0, None)]*12
# # # # # #     res = linprog(
# # # # # #         cost,
# # # # # #         A_eq=A[:6], b_eq=b[:6],
# # # # # #         A_ub=A[6:], b_ub=b[6:],
# # # # # #         bounds=bounds,
# # # # # #         method='highs',
# # # # # #         options={'presolve': True,
# # # # # #                  'primal_feasibility_tolerance': 1e-12,
# # # # # #                  'dual_feasibility_tolerance': 1e-12
# # # # # #                  }
# # # # # #     )
# # # # # #     # 返回吨千米数和分配方案
# # # # # #     return res.fun, res.x
# # # # # #
# # # # # # # 优化目标只用吨千米数
# # # # # # def obj(depot_coords):
# # # # # #     tkm, _ = total_tkm(depot_coords)
# # # # # #     return tkm
# # # # # #
# # # # # # x0 = np.array([5,1,2,7])
# # # # # # result = minimize(obj, x0, method='Nelder-Mead', options={'xatol':1e-12, 'fatol':1e-12, 'maxiter':10000})
# # # # # #
# # # # # # # 得到最优分配方案
# # # # # # tkm, allocation = total_tkm(result.x)
# # # # # # print("新料场坐标：", np.round(result.x.reshape(2,2), 8))
# # # # # # print("最小吨千米数：", tkm)
# # # # # # print("分配方案（A1~A6, B1~B6）：", allocation)
# # # # #
# # # # # import numpy as np
# # # # # from scipy.optimize import minimize, linprog
# # # # #
# # # # # sites = np.array([
# # # # #     [1.25, 1.25],
# # # # #     [8.75, 0.75],
# # # # #     [0.5, 4.75],
# # # # #     [5.75, 5],
# # # # #     [3, 6.5],
# # # # #     [7.25, 7.25]
# # # # # ])
# # # # # demands = np.array([3, 5, 4, 7, 6, 11])
# # # # # supplies = np.array([20, 20])
# # # # #
# # # # # def total_tkm(depot_coords):
# # # # #     depots = depot_coords.reshape(2,2)
# # # # #     dist = np.linalg.norm(depots[:, None, :] - sites[None, :, :], axis=2)
# # # # #     cost = dist.flatten()
# # # # #     A = []
# # # # #     for j in range(6):
# # # # #         row = [0]*12
# # # # #         row[j] = 1
# # # # #         row[6+j] = 1
# # # # #         A.append(row)
# # # # #     b = demands.tolist()
# # # # #     for i in range(2):
# # # # #         row = [0]*12
# # # # #         for j in range(6):
# # # # #             row[i*6 + j] = 1
# # # # #         A.append(row)
# # # # #     b += supplies.tolist()
# # # # #     bounds = [(0, None)]*12
# # # # #
# # # # #     # 提高HiGHS求解精度
# # # # #     res = linprog(
# # # # #         cost,
# # # # #         A_eq=A[:6], b_eq=b[:6],
# # # # #         A_ub=A[6:], b_ub=b[6:],
# # # # #         bounds=bounds,
# # # # #         method='highs',
# # # # #         options={
# # # # #             'presolve': True,
# # # # #             'primal_feasibility_tolerance': 1e-12,
# # # # #             'dual_feasibility_tolerance': 1e-12
# # # # #         }
# # # # #     )
# # # # #     return res.fun, res.x
# # # # #
# # # # # def obj(depot_coords):
# # # # #     tkm, _ = total_tkm(depot_coords)
# # # # #     return tkm
# # # # #
# # # # # x0 = np.array([5,1,2,7])
# # # # # # 提高坐标优化精度
# # # # # result = minimize(obj, x0, method='Nelder-Mead',
# # # # #                  options={'xatol':1e-12, 'fatol':1e-12, 'maxiter':10000})
# # # # #
# # # # # tkm, allocation = total_tkm(result.x)
# # # # # print("新料场坐标：", np.round(result.x.reshape(2,2), 8))
# # # # # print("最小吨千米数：", np.round(tkm, 8))
# # # # # print("分配方案（A1~A6, B1~B6）：", np.round(allocation, 8))
# # # #
# # # # import numpy as np
# # # # from scipy.optimize import minimize, linprog
# # # #
# # # # sites = np.array([
# # # #     [1.25, 1.25],
# # # #     [8.75, 0.75],
# # # #     [0.5, 4.75],
# # # #     [5.75, 5],
# # # #     [3, 6.5],
# # # #     [7.25, 7.25]
# # # # ])
# # # # demands = np.array([3, 5, 4, 7, 6, 11])
# # # # supplies = np.array([20, 20])
# # # #
# # # # def total_tkm(depot_coords):
# # # #     depots = depot_coords.reshape(2,2)
# # # #     dist = np.linalg.norm(depots[:, None, :] - sites[None, :, :], axis=2)
# # # #     cost = dist.flatten()
# # # #     A = []
# # # #     for j in range(6):
# # # #         row = [0]*12
# # # #         row[j] = 1
# # # #         row[6+j] = 1
# # # #         A.append(row)
# # # #     b = demands.tolist()
# # # #     for i in range(2):
# # # #         row = [0]*12
# # # #         for j in range(6):
# # # #             row[i*6 + j] = 1
# # # #         A.append(row)
# # # #     b += supplies.tolist()
# # # #     bounds = [(0, None)]*12
# # # #
# # # #     # 使用interior-point方法获得连续解
# # # #     res = linprog(
# # # #         cost,
# # # #         A_eq=A[:6], b_eq=b[:6],
# # # #         A_ub=A[6:], b_ub=b[6:],
# # # #         bounds=bounds,
# # # #         method='interior-point',  # 改用interior-point方法
# # # #         options={'sparse': True}
# # # #     )
# # # #     return res.fun, res.x
# # # #
# # # # def obj(depot_coords):
# # # #     tkm, _ = total_tkm(depot_coords)
# # # #     return tkm
# # # #
# # # # x0 = np.array([5,1,2,7])
# # # # result = minimize(obj, x0, method='Nelder-Mead',
# # # #                  options={'xatol':1e-12, 'fatol':1e-12, 'maxiter':10000})
# # # #
# # # # tkm, allocation = total_tkm(result.x)
# # # # print("新料场坐标：", np.round(result.x.reshape(2,2), 8))
# # # # print("最小吨千米数：", np.round(tkm, 8))
# # # # print("分配方案（A1~A6, B1~B6）：", np.round(allocation, 8))
# # #
# # # import numpy as np
# # # from scipy.optimize import minimize, linprog
# # #
# # # sites = np.array([
# # #     [1.25, 1.25],
# # #     [8.75, 0.75],
# # #     [0.5, 4.75],
# # #     [5.75, 5],
# # #     [3, 6.5],
# # #     [7.25, 7.25]
# # # ])
# # # demands = np.array([3, 5, 4, 7, 6, 11])
# # # supplies = np.array([20, 20])
# # #
# # # def total_tkm(depot_coords):
# # #     depots = depot_coords.reshape(2,2)
# # #     dist = np.linalg.norm(depots[:, None, :] - sites[None, :, :], axis=2)
# # #     cost = dist.flatten()
# # #
# # #     # 添加微小随机扰动避免整数解
# # #     np.random.seed(42)  # 固定随机种子保证结果可重现
# # #     cost += np.random.normal(0, 1e-10, size=cost.shape)
# # #
# # #     A = []
# # #     for j in range(6):
# # #         row = [0]*12
# # #         row[j] = 1
# # #         row[6+j] = 1
# # #         A.append(row)
# # #     b = demands.tolist()
# # #     for i in range(2):
# # #         row = [0]*12
# # #         for j in range(6):
# # #             row[i*6 + j] = 1
# # #         A.append(row)
# # #     b += supplies.tolist()
# # #     bounds = [(0, None)]*12
# # #
# # #     res = linprog(
# # #         cost,
# # #         A_eq=A[:6], b_eq=b[:6],
# # #         A_ub=A[6:], b_ub=b[6:],
# # #         bounds=bounds,
# # #         method='highs',
# # #         options={'presolve': True}
# # #     )
# # #     return res.fun, res.x
# # #
# # # def obj(depot_coords):
# # #     tkm, _ = total_tkm(depot_coords)
# # #     return tkm
# # #
# # # x0 = np.array([5,1,2,7])
# # # result = minimize(obj, x0, method='Nelder-Mead',
# # #                  options={'xatol':1e-12, 'fatol':1e-12, 'maxiter':10000})
# # #
# # # tkm, allocation = total_tkm(result.x)
# # # print("新料场坐标：", np.round(result.x.reshape(2,2), 8))
# # # print("最小吨千米数：", np.round(tkm, 8))
# # # print("分配方案（A1~A6, B1~B6）：", np.round(allocation, 8))
# #
# #
# #
# # # 3. 问题 2：改建新料场（演示聚类 + 重心法思路）
# # from scipy.cluster.vq import kmeans, vq
# #
# # # 先将工地按需求量加权聚类（K=2）
# # # 构造加权坐标（需求量为权重）
# # weighted_coords = np.array([(demand[i] * site_x[i], demand[i] * site_y[i]) for i in range(n_sites)])
# # sum_weights = np.sum(demand)
# # # K-means 聚类（按坐标，后续用需求加权修正）
# # centroids, _ = kmeans(sites[:, :2], 2)
# # # 分配工地到聚类
# # labels, _ = vq(sites[:, :2], centroids)
# #
# # # 对每个聚类用重心法求新料场坐标
# # new_sites = []
# # for label in [0, 1]:
# #     mask = labels == label
# #     cluster_demand = demand[mask]
# #     cluster_x = site_x[mask]
# #     cluster_y = site_y[mask]
# #     # 重心坐标：(sum(demand_i * x_i)/sum_demand, sum(demand_i * y_i)/sum_demand)
# #     cx = np.sum(cluster_demand * cluster_x) / np.sum(cluster_demand)
# #     cy = np.sum(cluster_demand * cluster_y) / np.sum(cluster_demand)
# #     new_sites.append([cx, cy])
# # new_sites = np.array(new_sites)
# #
# # # 计算改建后吨千米数（假设新料场按重心，运输量按需求分配）
# # def calc_new_cost(new_sites):
# #     total = 0
# #     for i in range(2):
# #         dist = np.sqrt((new_sites[i, 0] - site_x)**2 + (new_sites[i, 1] - site_y)**2)
# #         # 简单分配：一个料场供一个聚类（实际需优化运输量，这里演示）
# #         mask = labels == i
# #         total += np.sum(demand[mask] * dist[mask])
# #     return total
# #
# # old_cost = res_nonlin.fun  # 问题 1 最优解作为改建前成本
# # new_cost = calc_new_cost(new_sites)
# # save = old_cost - new_cost
# #
# # print("建议新料场坐标：", new_sites)
# # print("节省吨千米数：", save)
# #
# # # 若要更精确，可将新料场坐标作为变量，与运输量一起优化（扩展问题 1 的非线性模型，把料场坐标也放入变量）
#
#
# import numpy as np
# from scipy.optimize import linprog, minimize
#
# # 1. 数据准备
# # 料场坐标
# A = np.array([5, 1])
# B = np.array([2, 7])
# # 工地坐标 (x, y) 及需求量
# sites = np.array([
#     [1.25, 1.25, 3],
#     [8.75, 0.75, 5],
#     [0.5, 4.75, 4],
#     [5.75, 5, 7],
#     [3, 6.5, 6],
#     [7.25, 7.25, 11]
# ])
# site_x = sites[:, 0]
# site_y = sites[:, 1]
# demand = sites[:, 2]
# n_sites = len(demand)
#
# # 2. 问题 1：线性近似（距离用曼哈顿距离简化，或直接用欧氏距离线性化）
# # 这里先演示曼哈顿距离近似（系数矩阵构造）
# # 目标函数系数：料场 A 到各工地距离（x 差绝对值 + y 差绝对值）* 运输量系数（1 吨对应距离）
# cost_A = np.abs(A[0] - site_x) + np.abs(A[1] - site_y)
# cost_B = np.abs(B[0] - site_x) + np.abs(B[1] - site_y)
# # 构造目标函数系数：x_A1, x_A2,...x_A6, x_B1,...x_B6
# c = np.concatenate([cost_A, cost_B])
#
# # 约束条件：
# # 需求约束：x_Ai + x_Bi = demand[i]
# eq_cons = []
# for i in range(n_sites):
#     row = np.zeros(2 * n_sites)
#     row[i] = 1
#     row[i + n_sites] = 1
#     eq_cons.append(row)
# eq_cons = np.array(eq_cons)
# eq_rhs = demand
#
# # 料场储量约束：sum(x_Ai) <= 20, sum(x_Bi) <= 20
# ineq_cons = np.array([
#     [1] * n_sites + [0] * n_sites,
#     [0] * n_sites + [1] * n_sites
# ])
# ineq_rhs = [20, 20]
#
# # 线性规划求解（因实际是非线性，这里仅演示线性近似，若要精确需用非线性方法）
# # 线性规划只能处理 <= 约束，等式约束放参数中
# res_lin = linprog(c, A_ub=ineq_cons, b_ub=ineq_rhs, A_eq=eq_cons, b_eq=eq_rhs, bounds=(0, None))
#
# # 非线性精确求解（用 minimize 处理欧氏距离）
# def objective_nonlin(x):
#     x_A = x[:n_sites]
#     x_B = x[n_sites:]
#     dist_A = np.sqrt((A[0] - site_x)**2 + (A[1] - site_y)**2)
#     dist_B = np.sqrt((B[0] - site_x)**2 + (B[1] - site_y)**2)
#     return np.sum(x_A * dist_A + x_B * dist_B)
#
# # 初始值用线性规划结果
# x0 = res_lin.x
# bounds = [(0, None)] * (2 * n_sites)
# # 等式约束：x_Ai + x_Bi = demand[i]
# cons = [{'type': 'eq', 'fun': lambda x, i=i: x[i] + x[i + n_sites] - demand[i]} for i in range(n_sites)]
# # 不等式约束：sum(x_A) <=20, sum(x_B)<=20
# cons.append({'type': 'ineq', 'fun': lambda x: 20 - np.sum(x[:n_sites])})
# cons.append({'type': 'ineq', 'fun': lambda x: 20 - np.sum(x[n_sites:])})
#
# res_nonlin = minimize(objective_nonlin, x0, bounds=bounds, constraints=cons)
#
# print("问题 1 线性近似解（运输量）：", res_lin.x)
# print("问题 1 非线性精确解（运输量）：", res_nonlin.x)
# print("问题 1 非线性目标函数值（吨千米数）：", res_nonlin.fun)
#
#
# # 3. 问题 2：改建新料场（演示聚类 + 重心法思路）
# from scipy.cluster.vq import kmeans, vq
#
# # 先将工地按需求量加权聚类（K=2）
# # 构造加权坐标（需求量为权重）
# weighted_coords = np.array([(demand[i] * site_x[i], demand[i] * site_y[i]) for i in range(n_sites)])
# sum_weights = np.sum(demand)
# # K-means 聚类（按坐标，后续用需求加权修正）
# centroids, _ = kmeans(sites[:, :2], 2)
# # 分配工地到聚类
# labels, _ = vq(sites[:, :2], centroids)
#
# # 对每个聚类用重心法求新料场坐标
# new_sites = []
# for label in [0, 1]:
#     mask = labels == label
#     cluster_demand = demand[mask]
#     cluster_x = site_x[mask]
#     cluster_y = site_y[mask]
#     # 重心坐标：(sum(demand_i * x_i)/sum_demand, sum(demand_i * y_i)/sum_demand)
#     cx = np.sum(cluster_demand * cluster_x) / np.sum(cluster_demand)
#     cy = np.sum(cluster_demand * cluster_y) / np.sum(cluster_demand)
#     new_sites.append([cx, cy])
# new_sites = np.array(new_sites)
#
# # 计算改建后吨千米数（假设新料场按重心，运输量按需求分配）
# def calc_new_cost(new_sites):
#     total = 0
#     for i in range(2):
#         dist = np.sqrt((new_sites[i, 0] - site_x)**2 + (new_sites[i, 1] - site_y)**2)
#         # 简单分配：一个料场供一个聚类（实际需优化运输量，这里演示）
#         mask = labels == i
#         total += np.sum(demand[mask] * dist[mask])
#     return total
#
# old_cost = res_nonlin.fun  # 问题 1 最优解作为改建前成本
# new_cost = calc_new_cost(new_sites)
# save = old_cost - new_cost
#
# print("建议新料场坐标：", new_sites)
# print("节省吨千米数：", save)
#
# # 若要更精确，可将新料场坐标作为变量，与运输量一起优化（扩展问题 1 的非线性模型，把料场坐标也放入变量）
# #

import numpy as np
from scipy.optimize import minimize, linprog

sites = np.array([
    [1.25, 1.25],
    [8.75, 0.75],
    [0.5, 4.75],
    [5.75, 5],
    [3, 6.5],
    [7.25, 7.25]
])
demands = np.array([3, 5, 4, 7, 6, 11])
supplies = np.array([20, 20])

# 预生成固定的随机扰动
np.random.seed(42)
cost_perturbation = np.random.normal(0, 1e-12, size=12)  # 更小的扰动

def total_tkm(depot_coords):
    depots = depot_coords.reshape(2,2)
    dist = np.linalg.norm(depots[:, None, :] - sites[None, :, :], axis=2)
    cost = dist.flatten() + cost_perturbation  # 使用固定扰动

    A = []
    for j in range(6):
        row = [0]*12
        row[j] = 1
        row[6+j] = 1
        A.append(row)
    b = demands.tolist()
    for i in range(2):
        row = [0]*12
        for j in range(6):
            row[i*6 + j] = 1
        A.append(row)
    b += supplies.tolist()
    bounds = [(0, None)]*12

    res = linprog(
        cost,
        A_eq=A[:6], b_eq=b[:6],
        A_ub=A[6:], b_ub=b[6:],
        bounds=bounds,
        method='highs',
        options={'presolve': True}
    )
    return res.fun, res.x

def obj(depot_coords):
    tkm, _ = total_tkm(depot_coords)
    return tkm

x0 = np.array([5,1,2,7])
result = minimize(obj, x0, method='Nelder-Mead',
                 options={'xatol':1e-12, 'fatol':1e-12, 'maxiter':10000})

tkm, allocation = total_tkm(result.x)
print("新料场坐标：", np.round(result.x.reshape(2,2), 8))
print("最小吨千米数：", np.round(tkm, 8))
print("分配方案（A1~A6, B1~B6）：", np.round(allocation, 8))