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
    def __init__(self, start_point: str, speed: str, capacity: int):
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
        optimal_path = self.reconstruct_path(start_mask, start_node)
        path = ["n" + str(i) for i in optimal_path]
        path.insert(0, "r0")
        path.insert(len(path), "r0")
        print(path)
        print("Minimum Cost:", min_cost)

        # Ensure all neighborhoods are covered
        all_neighborhoods = set(range(self.num_nodes))
        covered_neighborhoods = set(optimal_path)
        missing_neighborhoods = all_neighborhoods - covered_neighborhoods

        if missing_neighborhoods:
            print(f"Warning: The optimal path does not cover all neighborhoods. Missing neighborhoods: {missing_neighborhoods}")

        return optimal_path

def generate_delivery_schedule(data: SaibabaColonyData):
    # Assuming only one restaurant and one vehicle for simplicity
    restaurant = data.restaurants["r0"]
    vehicle = data.vehicles["v0"]

    # Get the optimal TSP path
    tsp_path = get_optimal_tsp_path(data)

    # Initialize variables for delivery schedule
    delivery_schedule = []
    current_capacity = 0
    current_distance = 0
    current_slot = []

    for node in tsp_path:
        if node.startswith("n"):
            neighborhood = data.neighbourhoods[node]
            order_quantity = neighborhood.order_quantity

            # Check if the order can fit in the vehicle's capacity
            if current_capacity + order_quantity <= vehicle.capacity:
                current_capacity += order_quantity
                current_slot.append(node)
                current_distance += neighborhood.distances[int(tsp_path[0][1:])]  # Distance to starting node
            else:
                # Start a new delivery slot
                delivery_schedule.append({
                    "slot": current_slot,
                    "capacity": current_capacity,
                    "distance": current_distance
                })
                current_slot = [node]
                current_capacity = order_quantity
                current_distance = neighborhood.distances[int(tsp_path[0][1:])]

    # Add the last slot to the delivery schedule
    if current_slot:
        delivery_schedule.append({
            "slot": current_slot,
            "capacity": current_capacity,
            "distance": current_distance
        })

    return delivery_schedule


def get_optimal_tsp_path(data: SaibabaColonyData):
    n_neighbourhoods = data.n_neighbourhoods
    distance_matrix_2d = np.zeros((n_neighbourhoods, n_neighbourhoods))  # Fix the typo

    # Fill the distance matrix
    for key, entry in data.neighbourhoods.items():
        distances = entry.distances
        index = int(key[1:])  # Extract the index from the key (e.g., "n0" -> 0)
        distance_matrix_2d[index] = distances

    # Run the Held-Karp algorithm
    held_karp_instance = HeldKarp(distance_matrix_2d)
    optimal_path = held_karp_instance.run()

    if optimal_path:
        # Return the optimal TSP path
        start_node = 0
        tsp_path = ["n" + str(i) for i in optimal_path]
        tsp_path.insert(0, "r0")
        tsp_path.insert(len(tsp_path), "r0")
        return tsp_path
    else:
        print("Error: Unable to find an optimal path.")
        return []


def format_delivery_schedule(delivery_schedule, selected_paths):
    formatted_schedule = {}
    vehicle_id = "v0"
    paths = {}

    for i, slot in enumerate(delivery_schedule):
        path_key = f"path{i + 1}"
        if path_key in selected_paths:
            path_value = ["r0"] + slot["slot"] + ["r0"]
            paths[path_key] = path_value

    formatted_schedule[vehicle_id] = paths
    return formatted_schedule


# Sample input data
saibaba_colony_data = SaibabaColonyData(
    n_neighbourhoods=20,
    n_restaurants=1,
    neighbourhoods={
        "n0": Neighborhood(order_quantity=70, distances=[0, 2953, 1170, 1677, 1318, 2055, 591, 3050, 2626, 1864, 277, 2499, 769, 1463, 2006, 2516, 2394, 997, 1099, 421]),
        "n1": Neighborhood(order_quantity=70, distances=[2953, 0, 1783, 1276, 1635, 898, 2458, 97, 423, 1089, 3026, 664, 2280, 1600, 1057, 535, 559, 2182, 2208, 2532]),
        "n2": Neighborhood(order_quantity=90, distances=[1170, 1783, 0, 507, 148, 885, 675, 1880, 1456, 694, 1447, 1697, 497, 2633, 2090, 1346, 1224, 2167, 2269, 953]),
        "n3": Neighborhood(order_quantity=50, distances=[1677, 1276, 507, 0, 359, 752, 1182, 1373, 1325, 187, 1750, 1566, 1004, 2502, 1959, 839, 717, 2036, 2138, 1256]),
        "n4": Neighborhood(order_quantity=70, distances=[1318, 1635, 148, 359, 0, 737, 823, 1732, 1310, 546, 1391, 1551, 645, 2487, 1944, 1198, 1076, 2021, 2123, 897]),
        "n5": Neighborhood(order_quantity=90, distances=[2055, 898, 885, 752, 737, 0, 1560, 995, 573, 939, 2128, 814, 1382, 1750, 1207, 461, 339, 1284, 1386, 1634]),
        "n6": Neighborhood(order_quantity=110, distances=[591, 2458, 675, 1182, 823, 1560, 0, 2555, 2131, 1369, 868, 2004, 178, 2054, 1511, 2021, 1899, 1588, 1690, 374]),
        "n7": Neighborhood(order_quantity=70, distances=[3050, 97, 1880, 1373, 1732, 995, 2555, 0, 424, 1186, 3123, 665, 2377, 1601, 1058, 534, 656, 2279, 2305, 2629]),
        "n8": Neighborhood(order_quantity=110, distances=[2626, 423, 1456, 1325, 1310, 573, 2131, 424, 0, 1512, 2699, 241, 1953, 1177, 634, 958, 608, 1855, 1881, 2205]),
        "n9": Neighborhood(order_quantity=70, distances=[1864, 1089, 694, 187, 546, 939, 1369, 1186, 1512, 0, 1937, 1753, 1191, 2689, 2146, 652, 904, 2223, 2325, 1443]),
        "n10": Neighborhood(order_quantity=70, distances=[277, 3026, 1447, 1750, 1391, 2128, 868, 3123, 2699, 1937, 0, 2572, 1046, 1536, 2079, 2589, 2467, 844, 822, 494]),
        "n11": Neighborhood(order_quantity=110, distances=[2499, 664, 1697, 1566, 1551, 814, 2004, 665, 241, 1753, 2572, 0, 1826, 1036, 493, 1199, 849, 1728, 1754, 2078]),
        "n12": Neighborhood(order_quantity=110, distances=[769, 2280, 497, 1004, 645, 1382, 178, 2377, 1953, 1191, 1046, 1826, 0, 2232, 1689, 1843, 1721, 1766, 1868, 552]),
        "n13": Neighborhood(order_quantity=90, distances=[1463, 1600, 2633, 2502, 2487, 1750, 2054, 1601, 1177, 2689, 1536, 1036, 2232, 0, 543, 2135, 1785, 692, 718, 1680]),
        "n14": Neighborhood(order_quantity=50, distances=[2006, 1057, 2090, 1959, 1944, 1207, 1511, 1058, 634, 2146, 2079, 493, 1689, 543, 0, 1592, 1242, 1235, 1261, 1585]),
        "n15": Neighborhood(order_quantity=90, distances=[2516, 535, 1346, 839, 1198, 461, 2021, 534, 958, 652, 2589, 1199, 1843, 2135, 1592, 0, 350, 1745, 1771, 2095]),
        "n16": Neighborhood(order_quantity=110, distances=[2394, 559, 1224, 717, 1076, 339, 1899, 656, 608, 904, 2467, 849, 1721, 1785, 1242, 350, 0, 1623, 1649, 1973]),
        "n17": Neighborhood(order_quantity=90, distances=[997, 2182, 2167, 2036, 2021, 1284, 1588, 2279, 1855, 2223, 844, 1728, 1766, 692, 1235, 1745, 1623, 0, 102, 1214]),
        "n18": Neighborhood(order_quantity=70, distances=[1099, 2208, 2269, 2138, 2123, 1386, 1690, 2305, 1881, 2325, 822, 1754, 1868,718,1261,1771,1649,102,0,1316]),
        "n19": Neighborhood(order_quantity=110, distances=[421, 2532, 953, 1256, 897, 1634, 374, 2629, 2205, 1443, 494, 2078, 552, 1680, 1585, 2095, 1973, 1214, 1316, 0])

        # Add neighbourhood data
    },
    restaurants={
        "r0": Restaurant(neighbourhood_distance=[797, 2156, 563, 880, 521, 1258, 302, 2253, 1829, 1067, 884, 1702, 162, 2070, 1527, 1719, 1597, 1604, 1706, 390],
                        restaurant_distance=[0]),
        # Add restaurant data
    },
    vehicles={"v0": Vehicle(start_point="r0", speed="INF", capacity=600)}
)


delivery_schedule = generate_delivery_schedule(saibaba_colony_data)
selected_paths = {"path1", "path2", "path3"}  # Add or remove paths as needed
formatted_schedule = format_delivery_schedule(delivery_schedule, selected_paths)
print("Formatted Delivery Schedule:")
print(json.dumps(formatted_schedule, indent=2))
json_string = json.dumps(formatted_schedule, indent=2)
print(json_string)
with open('level1a_output.json', 'w') as f:
    json.dump(formatted_schedule, f)
    