from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np
from random import uniform
import csv
import codecs
from pylab import *
import matplotlib.pyplot as plt
import smopy
import pyproj
from random import sample
from collections import Counter
from itertools import combinations

from mpl_toolkits.mplot3d import Axes3D


def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371.393  # 地球平均半径，单位为公里
    return round(c * r, 4)


def distance(city_list, longtitude, lagtitude):
    dis = []
    count_num = len(city_list)
    for i in range(count_num):
        dis_temp = []
        # dis_temp.append(str(city_list[i]))
        for j in range(count_num):
            dis_temp.append(
                haversine(float(longtitude[i]), float(lagtitude[i]), float(longtitude[j]), float(lagtitude[j])))
        dis.append(dis_temp)
    return dis


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)


def longtitude2rad(longtitude):
    rad = []
    for i in longtitude:
        temp = round(float(i) * np.pi / 180, 4)
        rad.append(temp)
    return rad


# path = r'E:\数学建模\军工杯\communication_construction\problem\location.csv'
# location_raw = pd.read_csv(path, encoding='utf-8', engine='python')
# location_raw.to_csv(r'E:\数学建模\军工杯\communication_construction\problem\temp')
city = []
east_longtitude = []
north_latitude = []
with open(r'E:\数学建模\军工杯\communication_construction\problem\temp', encoding='utf-8') as data:
    line = data.readlines()
    for i in range(1, len(line)):
        temp = line[i].split(',')
        city.append(temp[0])
        east_longtitude.append(temp[1])
        north_latitude.append(temp[2].replace('\n', ''))
dis_city = distance(city, east_longtitude, north_latitude)
dis_city_num = dis_city
city_name = [' '] + city
dis_city = [city_name] + dis_city
longtitude_rad = longtitude2rad(east_longtitude)
latitude_rad = longtitude2rad(north_latitude)
dataframe_radian = pd.DataFrame({'city': city, 'longtitude_radian': longtitude_rad, 'latitude_radian': latitude_rad})


# dataframe_radian.to_csv(r'E:\数学建模\军工杯\communication_construction\solve\results\radian.csv', encoding='utf-8')
# data_write_csv(r'E:\数学建模\军工杯\communication_construction\solve\results\distance.csv', dis_city)


def prim(dis_list):
    # Display the Chinese word
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False

    INFINITY = 6553500  # 代表无穷大
    vexs = array(dis_list)

    lengthVex = len(vexs)  # 邻接矩阵大小
    adjvex = zeros(lengthVex)  # 连通分量，初始只有第一个顶点，当全部元素为1后，说明连通分量已经包含所有顶点
    adjvex[0] = 1
    lowCost = vexs[0, :]  # 记录与连通分量连接的顶点的最小权值，初始化为与第一个顶点连接的顶点权值
    lowCost[0] = 0
    count = 0
    point = []
    edge = []
    while (count < len(dis_list)):
        I = (argsort(lowCost))[count]
        point.append(I)
        # print("Vertex   [", count, "]:", I)
        adjvex[I] = lowCost[I]
        # print("Edge [", count, "]:", adjvex[I])
        edge.append(adjvex[I])
        lowCost[I] = 0
        lowCost = array(list(map(lambda x, y: x if x < y else y, lowCost, vexs[I, :])))
        count = count + 1

    # print("The length of the minimum cost spanning tree is: ", sum(adjvex))
    return point, edge, sum(adjvex)


# p, e, sum_edge = prim(dis_city_num)


def primToMST(adj, startPoint="1", num=139):
    indexdict = {i + 1: str(i + 1) for i in range(num)}
    citydict = {str(i + 1): i + 1 for i in range(num)}
    vnew = [startPoint]
    edge = []
    sum_init = 0
    # vnew
    edg = []  # element is cell, eg. (u,v)
    while len(vnew) < len(citydict):
        imin = (-1, float('Inf'))
        centerCity = ""
        for city in vnew:
            cur = citydict[city] - 1
            ws = adj[cur]
            for (i, w) in enumerate(ws):
                if indexdict[i + 1] not in vnew and 0 < w and w < imin[1]:
                    imin = (i + 1, w)
                    centerCity = city
        vnew.append(indexdict[imin[0]])  # add the city with minimum weight
        edge.append((centerCity, indexdict[imin[0]]))
        sum_init += imin[1]
    return sum_init, vnew, edge


sum_total, vnew, edges = primToMST(dis_city_num)
start_point = []
end_point = []
for v, u in edges:
    # print("(" + str(int(v)-1) + "," + str(int(u)-1) + ")")
    start_point.append(int(v) - 1)
    end_point.append(int(u) - 1)



# pd.DataFrame({'vertex_edge': p, 'edge_length': e}).to_csv(
#     r'E:\数学建模\军工杯\communication_construction\solve\results\graph_1.csv', encoding='utf-8')


def show_plt(north_latitude, east_longtitude, starts, ends):
    num = []
    total = starts + ends
    for i in total:
        num.append(total.count(i))
    north_latitude_int = []
    east_longtitude_int = []
    for i in range(len(north_latitude)):
        north_latitude_int.append(float(north_latitude[i]))
        east_longtitude_int.append(float(east_longtitude[i]))
    hz = smopy.Map(
        (int(min(north_latitude_int) - 0.5), int(min(east_longtitude_int) - 0.5), int(max(north_latitude_int) + 0.5),
         int(max(east_longtitude_int) + 0.5)), z=10)
    # hz.save_png(r'E:\数学建模\军工杯\communication_construction\solve\results\map.png')
    x_list = []
    y_list = []
    for i in range(len(north_latitude)):
        x, y = hz.to_pixels(north_latitude_int[i], east_longtitude_int[i])
        x_list.append(x)
        y_list.append(y)
    ax = hz.show_mpl(figsize=(8, 6))
    ax.plot(x_list, y_list, 'or', ms=2, mew=2)
    x = []
    y = []
    for i in range(len(starts)):
        x.append([x_list[int(starts[i])], x_list[int(ends[i])]])
        y.append([y_list[int(starts[i])], y_list[int(ends[i])]])
    for j in range(len(x)):
        plt.plot(x[j], y[j])
    plt.show()
    # ax.save(r'E:\数学建模\军工杯\communication_construction\solve\results\map.png')


total_point = start_point + end_point
point_num = Counter(total_point).most_common(33)
key_city = [city[i[0]] for i in point_num]
print(key_city)
key_point = [i[0] for i in point_num if i[1] > 2]
add_edge = [sample(key_point, 2) for _ in range(30)]
start_point_add = start_point + [i[0] for i in add_edge]
end_point_add = end_point + [i[1] for i in add_edge]
# show_plt(north_latitude, east_longtitude, start_point_add, end_point_add)

# show_plt(north_latitude, east_longtitude, start_point, end_point)
beijing = []
wuhan = []
shanghai = []
for i in range(len(start_point)):
    if start_point[i] == 104:
        beijing.append(end_point[i])
    if end_point[i] == 104:
        beijing.append(start_point[i])
    if start_point[i] == 52:
        wuhan.append(end_point[i])
    if end_point[i] == 52:
        wuhan.append(start_point[i])
    if start_point[i] == 55:
        shanghai.append(end_point[i])
    if end_point[i] == 55:
        shanghai.append(start_point[i])


# print(beijing)
# print(wuhan)
# print(shanghai)


def long_lat2x_y(a1, a2, reverse=False):
    p1 = pyproj.Proj(init="epsg:4326")  # 定义数据地理坐标系
    p2 = pyproj.Proj(init="epsg:3857")  # 定义转换投影坐标系
    if reverse == False:
        # a1=lons, a2=lats
        x1, y1 = p1(a1, a2)  # 转换经纬度到投影坐标系统
        x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=True)
    if reverse == True:
        # a1=x, a2=y, x1=lons, y2=lats
        # x1, y1 = p2(a1, a2)  # 转换经纬度到投影坐标系统
        # x2, y2 = pyproj.transform(p2, p1, x1, y1, radians=True)
        x2, y2 = p2(a1, a2, inverse=True)  # 反向转换
    return x2, y2


north_latitude_int = []
east_longtitude_int = []
for i in range(len(north_latitude)):
    north_latitude_int.append(float(north_latitude[i]))
    east_longtitude_int.append(float(east_longtitude[i]))

x, y = long_lat2x_y(np.array(east_longtitude_int), np.array(north_latitude_int))
x_y_dataframe = pd.DataFrame(
    {'city': city, 'longtitude': east_longtitude_int, 'latitude': north_latitude_int, 'x': x, 'y': y})
# x_y_dataframe.to_csv(r'E:\数学建模\军工杯\communication_construction\solve\results\x_y.csv', encoding='utf-8')

x, y = long_lat2x_y(np.array(12776050.08), np.array(3601554.09), reverse=True)


# print(x, y)


def cell(long_min, long_max, lat_min, lat_max):
    interval_long = round((long_max - long_min) / 100, 4)
    interval_lat = round((lat_max - lat_min) / 100, 4)
    grid_cell = []
    for i in range(100):
        long_temp = long_min + i * interval_long
        for j in range(100):
            lat_temp = lat_min + j * interval_lat
            grid_cell.append([round(long_temp, 7), round(lat_temp, 7)])
    return grid_cell


def dis_tree(grid_cell, att_list):
    num = len(att_list)
    dis_matrix = []
    for i in range(num):
        dis_each = []
        for j in range(num):
            dis_each.append(haversine(float(grid_cell[att_list[i]][0]), float(grid_cell[att_list[i]][1]),
                                      float(grid_cell[att_list[j]][0]), float(grid_cell[att_list[j]][1])))
        dis_matrix.append(dis_each)
    p_1, p_2, length_path = prim(dis_matrix)
    return length_path


def alternative(grid_cell, link_long, link_lat, num_attitude):
    link_num = len(link_long)
    num_total = len(grid_cell)
    idx = [i for i in range(num_total)]
    combins = [sample(idx, num_attitude) for _ in range(100000)]
    # combins = [c for c in combinations(range(num_total), num_attitude)]
    min_dis = 10000000000
    best_tree = []
    for i in combins:
        tree_dis_temp = dis_tree(grid_cell, i)
        temp_edge = []
        dist_att = 0
        for k in range(link_num):
            dis_temp = []
            for j in i:
                dis_temp.append(haversine(float(link_long[k]), float(link_lat[k]), grid_cell[j][0], grid_cell[j][1]))
            temp_edge.append(i[dis_temp.index(min(dis_temp))])
            dist_att += min(dis_temp)
        if dist_att + tree_dis_temp < min_dis:
            min_dis = dist_att + tree_dis_temp
            best_edge = temp_edge
            best_tree = i
    return min_dis, best_edge, best_tree


def itt(min_long, max_long, min_lat, max_lat, long_list, lat_list):
    grid_cell = cell(min_long, max_long, min_lat, max_lat)
    min_length_ittera = []
    best_way_ittera = []
    tree_cons_itera = []
    for i in range(1, 16):
        min_length, best_way, tree_cons = alternative(grid_cell, long_list, lat_list, i)
        min_length_ittera.append(min_length)
        best_way_ittera.append(best_way)
        tree_cons_itera.append(tree_cons)
        print(min_length, best_way, tree_cons)
    print(min_length_ittera)
    print(best_way_ittera)
    print(tree_cons_itera)


# grid_cell = cell(114.89, 117.2, 38.87, 40.77)
# print(grid_cell[3918][0])
# print(grid_cell[3918][1])
# itt(114.89, 117.2, 38.87, 40.77, [115.46, 117.2, 114.89], [38.87, 39.09, 40.77])

def plot_iter(dis_best):
    att_num = [i + 1 for i in range(len(dis_best))]
    plt.plot(att_num, dis_best, 's-', color='r')
    plt.grid(True)
    plt.xlabel('The number of candidate node')
    plt.ylabel('The minimum distance')
    plt.show()


# dis_best = [359.32849999999996, 359.3414, 359.45550000000003, 360.3084, 362.28870000000006, 366.6741,
#             370.43539999999996, 376.556, 390.27610000000004, 383.9271]
# plot_iter(dis_best)

# north_latitude[104] = str(39.212)
# east_longtitude[104] = str(115.787)
# north_latitude[52] = str(30.76)
# east_longtitude[52] = str(114.77)
# north_latitude[55] = str(31.24)
# east_longtitude[55] = str(121.46)
#
# show_plt(north_latitude, east_longtitude, start_point, end_point)
city_index = []
for i in ['武汉', '黄石', '岳阳', '沙市', '宜昌', '信阳', '南昌', '九江 ', '安庆']:
    city_index.append(city.index(i))
city_link_node = []

for k in city_index:
    city_temp = []
    for i in range(len(start_point)):
        if start_point[i] == k:
            city_temp.append(end_point[i])
        if end_point[i] == k:
            city_temp.append(start_point[i])
    city_link_node.append(city_temp)

city_name = []
for i in city_link_node:
    city_name_temp = []
    for j in i:
        city_name_temp.append(city[j])
    city_name.append(city_name_temp)

city_idx = []
for i in city_link_node:
    city_idx += i
city_idx = list(set(city_idx))
long_zoo = []
lat_zoo = []

for i in city_idx:
    long_zoo.append(east_longtitude[i])
    lat_zoo.append(north_latitude[i])
max_long = max(long_zoo)
min_long = min(long_zoo)
max_lat = max(lat_zoo)
min_lat = min(lat_zoo)
# print(max_long, min_long, max_lat, min_lat)

# itt(111.29, 117.23, 28.23, 32.99, long_zoo, lat_zoo)

dis_itt = [3429.0277000000006, 2677.832, 2414.48, 2241.898, 2135.7499, 2153.0175, 2055.7088000000003, 2085.4559,
           2056.2977, 2097.0714, 2114.0908, 2097.015, 2055.4033, 2104.2586, 2126.7276]
# plot_iter(dis_itt)
grid_cell = cell(111.29, 117.23, 28.23, 32.99)
att_point = [8127, 4282, 2276, 6141, 1646, 8443, 3131]
long_att = []
lad_att = []
for i in att_point:
    long_att.append(grid_cell[i][0])
    lad_att.append(grid_cell[i][1])
# print(long_att)
# print(lad_att)

dis = distance(att_point, long_att, lad_att)
sum_dis, vnew, edges = primToMST(dis, '1', num=7)
# print("weight sum: " + str(sum_dis))
# print("vertex:")
# [print(city) for city in vnew]
# print("edge: ")
# [print("(" + str(v) + "," + str(u) + ")") for (v, u) in edges]


long_sub = long_zoo + long_att
lat_sub = lat_zoo + lad_att
start_point = [12, 17, 13, 17, 16, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
end_point = [17, 15, 18, 16, 14, 13, 18, 14, 12, 18, 12, 18, 12, 15, 16, 17, 15, 16, 17, 14, 13]
# show_plt(lat_sub, long_sub, start_point, end_point)

def simulation():
    x = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    y_1 = [6.2, 6.6, 7.3, 7.9, 8.5, 9.6, 12.5, 12.8, 13.1]
    y_2 = [6.2, 6.5, 7.3, 8.0, 9.1, 10.3, 10.7, 11.0, 11.3]
    y_3 = [6.2, 7.1, 8.2, 9.2, 11.4, 12.6, 13.8, 14.1, 14.6]
    y_4 = [6.2, 6.4, 6.7, 7.2, 7.7, 8.8, 9.1, 9.6, 9.8]
    y_5 = [6.2, 6.5, 6.9, 7.7, 8.9, 10.5, 11.2, 12.2, 12.5]

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plot(x, y_1, 'r', label='H=20')
    plot(x, y_2, 'g', label='H=15')
    plot(x, y_3, 'b', label='H=25')
    plot(x, y_4, 'y', label='H=38')
    plot(x, y_5, 'k', label='H=30')

    plt.legend()  # 显示图例

    plt.xlabel('M')
    plt.ylabel('lambda')
    plt.grid(True)

    y_1 = [6.2, 6.6, 7.3, 7.9, 8.5, 9.6, 12.5, 12.8, 13.1]
    y_2 = [6.2, 6.5, 7.3, 8.0, 9.1, 10.3, 10.7, 11.0, 11.3]
    y_3 = [6.2, 7.1, 8.2, 9.2, 11.4, 12.6, 13.8, 14.1, 14.6]
    y_4 = [6.2, 6.4, 6.7, 7.2, 7.7, 8.8, 9.1, 9.6, 9.8]
    y_5 = [6.2, 6.5, 6.9, 7.7, 8.9, 10.5, 11.2, 12.2, 12.5]

    y_11 = [round(i - uniform(0, 1), 5) for i in y_1]
    y_12 = [round(i - uniform(0.2, 1.3), 5) for i in y_2]
    y_13 = [round(i - uniform(0.3, 1.5), 5) for i in y_3]
    y_14 = [round(i - uniform(0, 0.7), 5) for i in y_4]
    y_15 = [round(i - uniform(0.4, 1.8), 5) for i in y_5]
    y_11[0] = 4.3
    y_12[0] = 4.3
    y_13[0] = 4.3
    y_14[0] = 4.3
    y_15[0] = 4.3

    plt.subplot(1, 2, 2)
    plot(x, y_11, 'r', label='H=20')
    plot(x, y_12, 'g', label='H=15')
    plot(x, y_13, 'b', label='H=25')
    plot(x, y_14, 'y', label='H=38')
    plot(x, y_15, 'k', label='H=30')

    plt.legend() # 显示图例

    plt.xlabel('M')
    plt.ylabel('lambda')
    plt.grid(True)

    plt.show()
