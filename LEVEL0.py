import itertools
import math
import numpy as np
from typing import Dict, List
import json


class Neighborhood:
    def __init__(self, order_quantity: float, distances: List[float]):
        self.order_quantity = order_quantity
        self.distances = distances

class Restaurant:
    def __init__(self, neighbourhood_distance: List[float], restaurant_distance: List[float]):
        self.neighbourhood_distance = neighbourhood_distance
        self.restaurant_distance = restaurant_distance

class Vehicle:
    def __init__(self, start_point: str, speed: str, capacity: str):
        self.start_point = start_point
        self.speed = speed
        self.capacity = capacity

class SaibabaColonyData:
    def __init__(self, n_neighbourhoods: int, n_restaurants: int,
                 neighbourhoods: Dict[str, Neighborhood],
                 restaurants: Dict[str, Restaurant],
                 vehicles: Dict[str, Vehicle]):
        self.n_neighbourhoods = n_neighbourhoods
        self.n_restaurants = n_restaurants
        self.neighbourhoods = neighbourhoods
        self.restaurants = restaurants
        self.vehicles = vehicles

def run_tsp(data: SaibabaColonyData):
    n_neighbourhoods = data.n_neighbourhoods
    distance_matrix_2d = np.zeros((n_neighbourhoods, n_neighbourhoods))

    # Fill the distance matrix
    for key, entry in data.neighbourhoods.items():
        distances = entry.distances
        index = int(key[1:])  # Extract the index from the key (e.g., "n0" -> 0)
        distance_matrix_2d[index] = distances

    # Run the Held-Karp algorithm
    held_karp_instance = HeldKarp(distance_matrix_2d)
    held_karp_instance.run()

class HeldKarp:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.num_nodes = len(distance_matrix)
        self.memoization_table = {}

    def tsp_mask(self, mask, pos):
        if mask == (1 << self.num_nodes) - 1:
            return self.distance_matrix[pos][0]  # Return to the starting node

        if (mask, pos) in self.memoization_table:
            return self.memoization_table[(mask, pos)]

        min_cost = float('inf')

        for i in range(self.num_nodes):
            if not (mask & (1 << i)):
                new_mask = mask | (1 << i)
                cost = self.distance_matrix[pos][i] + self.tsp_mask(new_mask, i)

                if cost < min_cost:
                    min_cost = cost

        self.memoization_table[(mask, pos)] = min_cost
        return min_cost

   
    def reconstruct_path(self, mask, pos):
        path = [pos]
        visited = set([pos])

        while len(path) < self.num_nodes:
            next_node = min(
                (i for i in range(self.num_nodes) if i not in visited),
                key=lambda x: self.distance_matrix[pos][x] + self.tsp_mask(mask | (1 << x), x)
                )
            path.append(next_node)
            mask |= 1 << next_node
            pos = next_node
            visited.add(next_node)
            
        return path
    def run(self):
        start_node = 0
        start_mask = 1 << start_node
        min_cost = self.tsp_mask(start_mask, start_node)

        print("Optimal Path:")
        path1 = ["n"+str(i) for i in self.reconstruct_path(start_mask, start_node)]
        path1.insert(0,"r0")
        path1.insert(len(path1),"r0")
        print(path1)
        print("Minimum Cost:", min_cost)
        
        result={"v0":{"path":path1}}
        print(result)

        json_string = json.dumps(result, indent=2)
        print(json_string)
        with open('level0_output.json', 'w') as f:
            json.dump(result, f)


# Sample input data
saibaba_colony_data = SaibabaColonyData(
    n_neighbourhoods=20,
    n_restaurants=1,
    neighbourhoods={
        "n0": Neighborhood(order_quantity=float('inf'), distances=[0, 3366, 2290, 3118, 1345, 854, 1176, 1291, 1707, 2160, 1606, 702, 1820, 1985, 1838, 1515, 3370, 1643, 2874, 1418]),
        "n1": Neighborhood(order_quantity=float('inf'), distances=[3366, 0, 1076, 512, 2021, 2512, 2190, 2075, 1923, 1206, 1760, 2664, 1546, 1645, 1528, 1851, 376, 1723, 492, 1948]),
        "n2": Neighborhood(order_quantity=float('inf'), distances=[2290, 1076, 0, 1494, 945, 1436, 1114, 999, 2905, 536, 684, 1588, 876, 2627, 452, 775, 1358, 647, 716, 872]),
        "n3": Neighborhood(order_quantity=float('inf'), distances=[3118, 512, 1494, 0, 1773, 2264, 1942, 1827, 1411, 958, 1512, 2416, 1298, 1133, 1280, 1603, 252, 1475, 778, 1700]),
        "n4": Neighborhood(order_quantity=float('inf'), distances=[1345, 2021, 945, 1773, 0, 491, 403, 650, 2348, 815, 261, 787, 475, 2070, 493, 170, 2025, 298, 1529, 763]),
        "n5": Neighborhood(order_quantity=float('inf'), distances=[854, 2512, 1436, 2264, 491, 0, 322, 569, 2429, 1306, 752, 868, 966, 2151, 984, 661, 2516, 789, 2020, 682]),
        "n6": Neighborhood(order_quantity=float('inf'), distances=[1176, 2190, 1114, 1942, 403, 322, 0, 247, 2751, 984, 430, 1190, 722, 2473, 662, 521, 2194, 467, 1698, 360]),
        "n7": Neighborhood(order_quantity=float('inf'), distances=[1291, 2075, 999, 1827, 650, 569, 247, 0, 2998, 869, 677, 1437, 969, 2720, 547, 768, 2079, 352, 1583, 127]),
        "n8": Neighborhood(order_quantity=float('inf'), distances=[1707, 1923, 2905, 1411, 2348, 2429, 2751, 2998, 0, 2369, 2321, 1561, 2029, 278, 2553, 2230, 1663, 2646, 2189, 3111]),
        "n9": Neighborhood(order_quantity=float('inf'), distances=[2160, 1206, 536, 958, 815, 1306, 984, 869, 2369, 0, 554, 1458, 340, 2091, 322, 645, 1210, 517, 714, 742]),
        "n10": Neighborhood(order_quantity=float('inf'), distances=[1606, 1760, 684, 1512, 261, 752, 430, 677, 2321, 554, 0, 904, 292, 2043, 232, 91, 1764, 325, 1268, 790]),
        "n11": Neighborhood(order_quantity=float('inf'), distances=[702, 2664, 1588, 2416, 787, 868, 1190, 1437, 1561, 1458, 904, 0, 1118, 1283, 1136, 813, 2668, 1085, 2172, 1550]),
        "n12": Neighborhood(order_quantity=float('inf'), distances=[1820, 1546, 876, 1298, 475, 966, 722, 969, 2029, 340, 292, 1118, 0, 1751, 524, 305, 1550, 617, 1054, 1082]),
        "n13": Neighborhood(order_quantity=float('inf'), distances=[1985, 1645, 2627, 1133, 2070, 2151, 2473, 2720, 278, 2091, 2043, 1283, 1751, 0, 2275, 1952, 1385, 2368, 1911, 2833]),
        "n14": Neighborhood(order_quantity=float('inf'), distances=[1838, 1528, 452, 1280, 493, 984, 662, 547, 2553, 322, 232, 1136, 524, 2275, 0, 323, 1532, 195, 1036, 558]),
        "n15": Neighborhood(order_quantity=float('inf'), distances=[1515, 1851, 775, 1603, 170, 661, 521, 768, 2230, 645, 91, 813, 305, 1952, 323, 0, 1855, 416, 1359, 881]),
        "n16": Neighborhood(order_quantity=float('inf'), distances=[3370, 376, 1358, 252, 2025, 2516, 2194, 2079, 1663, 1210, 1764, 2668, 1550, 1385, 1532, 1855, 0, 1727, 642, 1952]),
        "n17": Neighborhood(order_quantity=float('inf'), distances=[1643, 1723, 647, 1475, 298, 789, 467, 352, 2646, 517, 325, 1085, 617, 2368, 195, 416, 1727, 0, 1231, 465]),
        "n18": Neighborhood(order_quantity=float('inf'), distances=[2874, 492, 716, 778, 1529, 2020, 1698, 1583, 2189, 714, 1268, 2172, 1054, 1911, 1036, 1359, 642, 1231, 0, 1456]),
        "n19": Neighborhood(order_quantity=float('inf'), distances=[1418, 1948, 872, 1700, 763, 682, 360, 127, 3111, 742, 790, 1550, 1082, 2833, 558, 881, 1952, 465, 1456, 0]),
    },
    restaurants={
        "r0": Restaurant(neighbourhood_distance=[2495, 1135, 2117, 623, 1560, 1641, 1963, 2210, 788, 1581, 1533, 1793, 1241, 510, 1765, 1442, 875, 1858, 1401, 2323], restaurant_distance=[0]),
        # Add other restaurants
    },
    vehicles={"v0": Vehicle(start_point="n0", speed="INF", capacity="INF")}
)

# Run the TSP algorithm
run_tsp(saibaba_colony_data)
