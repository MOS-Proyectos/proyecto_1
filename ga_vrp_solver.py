import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from copy import deepcopy
import random
from typing import List, Dict, Tuple

class Solution:
    def __init__(self, num_vehicles: int):
        self.routes = [[] for _ in range(num_vehicles)]  # One route per vehicle
        self.loads = [0] * num_vehicles    # Track load for each vehicle
        self.distances = [0] * num_vehicles  # Track distance for each vehicle
        
    def is_feasible(self, vehicles_data: pd.DataFrame) -> bool:
        """Check if solution respects capacity and range constraints."""
        for i, route in enumerate(self.routes):
            if self.loads[i] > vehicles_data.iloc[i]['Capacity'] or \
               self.distances[i] > vehicles_data.iloc[i]['Range']:
                return False
        return True
    
    def deep_copy(self):
        """Create a deep copy of the solution."""
        new_sol = Solution(len(self.routes))
        new_sol.routes = deepcopy(self.routes)
        new_sol.loads = deepcopy(self.loads)
        new_sol.distances = deepcopy(self.distances)
        return new_sol

class GAVRP:
    def __init__(self, 
                 depot_file: str,
                 clients_file: str,
                 vehicles_file: str,
                 population_size: int = 100,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 elitism_rate: float = 0.1):
        """
        Initialize the GA solver for Vehicle Routing Problem.
        
        Args:
            depot_file: Path to depot data CSV
            clients_file: Path to clients data CSV
            vehicles_file: Path to vehicles data CSV
            population_size: Size of GA population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            elitism_rate: Proportion of elite solutions to keep
        """
        # Load data
        self.depots = pd.read_csv(depot_file)
        self.clients = pd.read_csv(clients_file)
        self.vehicles = pd.read_csv(vehicles_file)
        
        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        
        # Initialize other attributes
        self.population: List[Solution] = []
        self.best_solution: Solution = None
        self.best_fitness: float = float('inf')
        
        # Precompute distance matrix
        self.distance_matrix = self._create_distance_matrix()
    
    def _haversine_distance(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calculate the Haversine distance between two points."""
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        R = 6371  # Radius of Earth in kilometers
        
        return R * c
    
    def _create_distance_matrix(self) -> np.ndarray:
        """Create a matrix of distances between all locations."""
        # Combine depot and client locations
        all_locations = pd.concat([
            self.depots[['LocationID', 'Longitude', 'Latitude']],
            self.clients[['LocationID', 'Longitude', 'Latitude']]
        ]).sort_values('LocationID')
        
        n = len(all_locations)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self._haversine_distance(
                    all_locations.iloc[i]['Longitude'],
                    all_locations.iloc[i]['Latitude'],
                    all_locations.iloc[j]['Longitude'],
                    all_locations.iloc[j]['Latitude']
                )
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist
                
        return dist_matrix
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance of a route including depot returns."""
        if not route:
            return 0
            
        total_distance = self.distance_matrix[0, route[0]-1]  # Depot to first client
        
        for i in range(len(route)-1):
            total_distance += self.distance_matrix[route[i]-1, route[i+1]-1]
            
        total_distance += self.distance_matrix[route[-1]-1, 0]  # Last client to depot
        
        return total_distance
    
    def create_initial_solution(self) -> Solution:
        """Create a single initial solution using a greedy randomized approach."""
        # Sort vehicles by capacity/range ratio for efficient assignment
        vehicle_indices = sorted(
            range(len(self.vehicles)),
            key=lambda i: self.vehicles.iloc[i]['Capacity'] / self.vehicles.iloc[i]['Range'],
            reverse=True
        )
        
        # Initialize solution
        solution = Solution(len(self.vehicles))
        unassigned = list(range(2, len(self.clients) + 2))  # Client LocationIDs
        
        for v_idx in vehicle_indices:
            vehicle = self.vehicles.iloc[v_idx]
            route = []
            current_load = 0
            
            while unassigned:
                feasible_clients = []
                
                for client in unassigned:
                    # Check capacity constraint
                    new_load = current_load + self.clients.iloc[client-2]['Demand']
                    if new_load > vehicle['Capacity']:
                        continue
                    
                    # Check range constraint
                    temp_route = route + [client]
                    new_distance = self.calculate_route_distance(temp_route)
                    if new_distance > vehicle['Range']:
                        continue
                    
                    feasible_clients.append(client)
                
                if not feasible_clients:
                    break
                
                # Select random client from feasible ones
                selected = random.choice(feasible_clients)
                route.append(selected)
                unassigned.remove(selected)
                current_load += self.clients.iloc[selected-2]['Demand']
            
            solution.routes[v_idx] = route
            solution.loads[v_idx] = current_load
            solution.distances[v_idx] = self.calculate_route_distance(route)
        
        return solution
    
    def initialize_population(self):
        """Initialize the population with random solutions."""
        self.population = []
        for _ in range(self.population_size):
            solution = self.create_initial_solution()
            self.population.append(solution) 
    
    def evaluate_fitness(self, solution: Solution) -> float:
        """
        Calculate the fitness of a solution with penalties for constraint violations.
        Lower fitness is better.
        """
        total_cost = 0
        penalties = 0
        
        # Calculate total distance and check constraints for each route
        for v_idx, route in enumerate(solution.routes):
            vehicle = self.vehicles.iloc[v_idx]
            
            # Distance cost
            route_distance = solution.distances[v_idx]
            total_cost += route_distance
            
            # Capacity violation penalty
            if solution.loads[v_idx] > vehicle['Capacity']:
                penalties += 1000 * (solution.loads[v_idx] - vehicle['Capacity'])
            
            # Range violation penalty
            if route_distance > vehicle['Range']:
                penalties += 1000 * (route_distance - vehicle['Range'])
            
            # Load balancing penalty (optional)
            if route:  # Only for non-empty routes
                load_ratio = solution.loads[v_idx] / vehicle['Capacity']
                penalties += 100 * abs(load_ratio - 0.8)  # Penalize underutilization
        
        # Penalty for unassigned clients
        assigned_clients = set(client for route in solution.routes for client in route)
        all_clients = set(range(2, len(self.clients) + 2))
        unassigned = all_clients - assigned_clients
        penalties += 10000 * len(unassigned)  # Heavy penalty for unassigned clients
        
        return total_cost + penalties
    
    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Perform crossover between two parent solutions."""
        if random.random() < 0.5:
            return self._route_exchange_crossover(parent1, parent2)
        else:
            return self._route_merge_crossover(parent1, parent2)
    
    def _route_exchange_crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Exchange complete routes between parents."""
        child1 = parent1.deep_copy()
        child2 = parent2.deep_copy()
        
        if len(self.vehicles) < 2:
            return child1, child2
        
        # Select random routes to exchange
        num_routes = random.randint(1, max(1, len(self.vehicles) // 2))
        routes_to_exchange = random.sample(range(len(self.vehicles)), num_routes)
        
        # Exchange the selected routes
        for route_idx in routes_to_exchange:
            # Swap routes
            child1.routes[route_idx], child2.routes[route_idx] = \
                child2.routes[route_idx], child1.routes[route_idx]
            
            # Update loads and distances
            child1.loads[route_idx] = sum(self.clients.iloc[c-2]['Demand'] 
                                        for c in child1.routes[route_idx])
            child2.loads[route_idx] = sum(self.clients.iloc[c-2]['Demand'] 
                                        for c in child2.routes[route_idx])
            
            child1.distances[route_idx] = self.calculate_route_distance(child1.routes[route_idx])
            child2.distances[route_idx] = self.calculate_route_distance(child2.routes[route_idx])
        
        return child1, child2
    
    def _route_merge_crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """Merge corresponding routes from parents."""
        child1 = Solution(len(self.vehicles))
        child2 = Solution(len(self.vehicles))
        
        # For each vehicle's route
        for v_idx in range(len(self.vehicles)):
            route1 = parent1.routes[v_idx]
            route2 = parent2.routes[v_idx]
            
            if route1 and route2:
                # Choose crossover points
                point1 = random.randint(0, len(route1))
                point2 = random.randint(0, len(route2))
                
                # Create new routes
                new_route1 = route1[:point1] + [x for x in route2[point2:] 
                                              if x not in route1[:point1]]
                new_route2 = route2[:point2] + [x for x in route1[point1:] 
                                              if x not in route2[:point2]]
                
                # Update children
                child1.routes[v_idx] = new_route1
                child2.routes[v_idx] = new_route2
                
                # Update loads and distances
                child1.loads[v_idx] = sum(self.clients.iloc[c-2]['Demand'] for c in new_route1)
                child2.loads[v_idx] = sum(self.clients.iloc[c-2]['Demand'] for c in new_route2)
                
                child1.distances[v_idx] = self.calculate_route_distance(new_route1)
                child2.distances[v_idx] = self.calculate_route_distance(new_route2)
        
        return child1, child2
    
    def mutate(self, solution: Solution) -> Solution:
        """Apply mutation operators to a solution."""
        mutated = solution.deep_copy()
        
        # Apply different mutation operators with certain probabilities
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(['swap', 'insert', 'inversion', 'redistribution'])
            
            if mutation_type == 'swap':
                mutated = self._swap_mutation(mutated)
            elif mutation_type == 'insert':
                mutated = self._insert_mutation(mutated)
            elif mutation_type == 'inversion':
                mutated = self._inversion_mutation(mutated)
            else:  # redistribution
                mutated = self._redistribution_mutation(mutated)
        
        return mutated
    
    def _swap_mutation(self, solution: Solution) -> Solution:
        """Swap two random clients within a route."""
        # Select a non-empty route
        non_empty_routes = [i for i, route in enumerate(solution.routes) if len(route) > 1]
        if not non_empty_routes:
            return solution
        
        route_idx = random.choice(non_empty_routes)
        route = solution.routes[route_idx]
        
        # Select two random positions
        pos1, pos2 = random.sample(range(len(route)), 2)
        
        # Swap clients
        route[pos1], route[pos2] = route[pos2], route[pos1]
        
        # Update distance
        solution.distances[route_idx] = self.calculate_route_distance(route)
        
        return solution
    
    def _insert_mutation(self, solution: Solution) -> Solution:
        """Move a client to a different position within its route."""
        # Select a non-empty route
        non_empty_routes = [i for i, route in enumerate(solution.routes) if len(route) > 1]
        if not non_empty_routes:
            return solution
        
        route_idx = random.choice(non_empty_routes)
        route = solution.routes[route_idx]
        
        # Select a random client and new position
        old_pos = random.randint(0, len(route) - 1)
        new_pos = random.randint(0, len(route) - 1)
        
        # Move client
        client = route.pop(old_pos)
        route.insert(new_pos, client)
        
        # Update distance
        solution.distances[route_idx] = self.calculate_route_distance(route)
        
        return solution
    
    def _inversion_mutation(self, solution: Solution) -> Solution:
        """Reverse a segment of a route."""
        # Select a non-empty route
        non_empty_routes = [i for i, route in enumerate(solution.routes) if len(route) > 1]
        if not non_empty_routes:
            return solution
        
        route_idx = random.choice(non_empty_routes)
        route = solution.routes[route_idx]
        
        # Select two positions
        pos1 = random.randint(0, len(route) - 2)
        pos2 = random.randint(pos1 + 1, len(route) - 1)
        
        # Reverse the segment
        route[pos1:pos2+1] = reversed(route[pos1:pos2+1])
        
        # Update distance
        solution.distances[route_idx] = self.calculate_route_distance(route)
        
        return solution
    
    def _redistribution_mutation(self, solution: Solution) -> Solution:
        """Move a client from one route to another."""
        # Find routes with clients
        non_empty_routes = [i for i, route in enumerate(solution.routes) if route]
        if len(non_empty_routes) < 2:
            return solution
        
        # Select source and destination routes
        from_idx = random.choice(non_empty_routes)
        to_idx = random.choice([i for i in range(len(solution.routes)) if i != from_idx])
        
        if not solution.routes[from_idx]:
            return solution
        
        # Select random client to move
        client_idx = random.randint(0, len(solution.routes[from_idx]) - 1)
        client = solution.routes[from_idx][client_idx]
        
        # Check if move is feasible
        new_load = solution.loads[to_idx] + self.clients.iloc[client-2]['Demand']
        if new_load <= self.vehicles.iloc[to_idx]['Capacity']:
            # Move client
            solution.routes[from_idx].pop(client_idx)
            solution.routes[to_idx].append(client)
            
            # Update loads
            solution.loads[from_idx] -= self.clients.iloc[client-2]['Demand']
            solution.loads[to_idx] = new_load
            
            # Update distances
            solution.distances[from_idx] = self.calculate_route_distance(solution.routes[from_idx])
            solution.distances[to_idx] = self.calculate_route_distance(solution.routes[to_idx])
        
        return solution 

    def evolve(self, verbose: bool = True) -> Solution:
        """
        Evolve the population for the specified number of generations.
        Returns the best solution found.
        """
        # Initialize population if not already done
        if not self.population:
            self.initialize_population()
        
        # Main evolution loop
        for generation in range(self.generations):
            # Evaluate current population
            population_fitness = [(i, self.evaluate_fitness(solution)) 
                                for i, solution in enumerate(self.population)]
            population_fitness.sort(key=lambda x: x[1])
            
            # Update best solution if needed
            current_best_fitness = population_fitness[0][1]
            current_best_solution = self.population[population_fitness[0][0]]
            
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = current_best_solution.deep_copy()
            
            if verbose and generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {self.best_fitness:.2f}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep best solutions
            num_elite = max(1, int(self.elitism_rate * self.population_size))
            elite_indices = [idx for idx, _ in population_fitness[:num_elite]]
            new_population.extend(self.population[idx].deep_copy() for idx in elite_indices)
            
            # Fill the rest of the population with offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population
            
            # Local search on best solution every 10 generations
            if generation % 10 == 0:
                self._local_search(self.best_solution)
        
        return self.best_solution
    
    def _tournament_selection(self, tournament_size: int = 5) -> Solution:
        """Select a parent using tournament selection."""
        # Randomly select tournament_size individuals
        tournament = random.sample(self.population, tournament_size)
        
        # Return the best one
        return min(tournament, key=lambda x: self.evaluate_fitness(x))
    
    def _local_search(self, solution: Solution) -> None:
        """Apply local search to improve a solution."""
        improved = True
        while improved:
            improved = False
            original_fitness = self.evaluate_fitness(solution)
            
            # Try different local search operators
            operators = [
                self._try_2opt,
                self._try_relocate,
                self._try_exchange
            ]
            
            for operator in operators:
                # Apply operator
                operator(solution)
                
                # Check if solution improved
                new_fitness = self.evaluate_fitness(solution)
                if new_fitness < original_fitness:
                    improved = True
                    original_fitness = new_fitness
    
    def _try_2opt(self, solution: Solution) -> None:
        """Try to improve routes using 2-opt local search."""
        for route_idx, route in enumerate(solution.routes):
            if len(route) < 3:
                continue
                
            improved = True
            while improved:
                improved = False
                best_distance = solution.distances[route_idx]
                
                for i in range(len(route) - 1):
                    for j in range(i + 2, len(route)):
                        # Try 2-opt swap
                        new_route = route[:i+1] + list(reversed(route[i+1:j+1])) + route[j+1:]
                        new_distance = self.calculate_route_distance(new_route)
                        
                        if new_distance < best_distance:
                            solution.routes[route_idx] = new_route
                            solution.distances[route_idx] = new_distance
                            best_distance = new_distance
                            improved = True
                            break
                    if improved:
                        break
    
    def _try_relocate(self, solution: Solution) -> None:
        """Try to relocate clients to better positions."""
        for from_route in range(len(solution.routes)):
            if not solution.routes[from_route]:
                continue
                
            for client_pos in range(len(solution.routes[from_route])):
                client = solution.routes[from_route][client_pos]
                best_increase = float('inf')
                best_move = None
                
                # Try all possible new positions
                for to_route in range(len(solution.routes)):
                    # Skip if same position in same route
                    if to_route == from_route and client_pos == len(solution.routes[to_route]):
                        continue
                    
                    # Check capacity constraint
                    if to_route != from_route:
                        new_load = solution.loads[to_route] + \
                                 self.clients.iloc[client-2]['Demand']
                        if new_load > self.vehicles.iloc[to_route]['Capacity']:
                            continue
                    
                    # Try insertion at each position
                    route = solution.routes[to_route]
                    for pos in range(len(route) + 1):
                        # Create new route
                        new_route = route[:pos] + [client] + route[pos:]
                        new_distance = self.calculate_route_distance(new_route)
                        
                        # Calculate cost increase
                        increase = new_distance - solution.distances[to_route]
                        if to_route == from_route:
                            # Subtract savings from removing client
                            old_route = solution.routes[from_route][:client_pos] + \
                                      solution.routes[from_route][client_pos+1:]
                            increase -= solution.distances[from_route] - \
                                      self.calculate_route_distance(old_route)
                        
                        if increase < best_increase:
                            best_increase = increase
                            best_move = (to_route, pos)
                
                # Apply best move if improvement found
                if best_move and best_increase < 0:
                    to_route, pos = best_move
                    # Remove from old route
                    solution.routes[from_route].pop(client_pos)
                    solution.loads[from_route] -= self.clients.iloc[client-2]['Demand']
                    solution.distances[from_route] = self.calculate_route_distance(
                        solution.routes[from_route]
                    )
                    
                    # Insert in new route
                    solution.routes[to_route].insert(pos, client)
                    solution.loads[to_route] += self.clients.iloc[client-2]['Demand']
                    solution.distances[to_route] = self.calculate_route_distance(
                        solution.routes[to_route]
                    )
    
    def _try_exchange(self, solution: Solution) -> None:
        """Try to exchange pairs of clients between routes."""
        for route1 in range(len(solution.routes)):
            if not solution.routes[route1]:
                continue
                
            for pos1 in range(len(solution.routes[route1])):
                client1 = solution.routes[route1][pos1]
                best_improvement = 0
                best_exchange = None
                
                for route2 in range(route1 + 1, len(solution.routes)):
                    if not solution.routes[route2]:
                        continue
                        
                    for pos2 in range(len(solution.routes[route2])):
                        client2 = solution.routes[route2][pos2]
                        
                        # Check capacity constraints
                        load1_delta = self.clients.iloc[client2-2]['Demand'] - \
                                    self.clients.iloc[client1-2]['Demand']
                        load2_delta = -load1_delta
                        
                        if (solution.loads[route1] + load1_delta > 
                            self.vehicles.iloc[route1]['Capacity'] or
                            solution.loads[route2] + load2_delta > 
                            self.vehicles.iloc[route2]['Capacity']):
                            continue
                        
                        # Try exchange
                        route1_copy = solution.routes[route1].copy()
                        route2_copy = solution.routes[route2].copy()
                        
                        route1_copy[pos1] = client2
                        route2_copy[pos2] = client1
                        
                        new_dist1 = self.calculate_route_distance(route1_copy)
                        new_dist2 = self.calculate_route_distance(route2_copy)
                        
                        improvement = (solution.distances[route1] + 
                                     solution.distances[route2] - 
                                     new_dist1 - new_dist2)
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_exchange = (route2, pos2)
                
                # Apply best exchange if improvement found
                if best_exchange and best_improvement > 0:
                    route2, pos2 = best_exchange
                    client2 = solution.routes[route2][pos2]
                    
                    # Perform exchange
                    solution.routes[route1][pos1], solution.routes[route2][pos2] = \
                        solution.routes[route2][pos2], solution.routes[route1][pos1]
                    
                    # Update loads
                    load_delta = self.clients.iloc[client2-2]['Demand'] - \
                               self.clients.iloc[client1-2]['Demand']
                    solution.loads[route1] += load_delta
                    solution.loads[route2] -= load_delta
                    
                    # Update distances
                    solution.distances[route1] = self.calculate_route_distance(
                        solution.routes[route1]
                    )
                    solution.distances[route2] = self.calculate_route_distance(
                        solution.routes[route2]
                    )
    
    def get_solution_summary(self, solution: Solution) -> dict:
        """Get a summary of the solution including routes, loads, and distances."""
        summary = {
            'total_distance': float(sum(solution.distances)),  # Convert to float
            'total_load': int(sum(solution.loads)),  # Convert to int
            'routes': [],
            'unassigned_clients': list(set(range(2, len(self.clients) + 2)) -  # Convert to list
                                set(client for route in solution.routes 
                                    for client in route))
        }
        
        for i, route in enumerate(solution.routes):
            summary['routes'].append({
                'vehicle_id': int(i + 1),  # Convert to int
                'route': [int(x) for x in route],  # Convert to list of ints
                'load': int(solution.loads[i]),  # Convert to int
                'distance': float(solution.distances[i]),  # Convert to float
                'capacity': int(self.vehicles.iloc[i]['Capacity']),  # Convert to int
                'range': float(self.vehicles.iloc[i]['Range'])  # Convert to float
            })
        
        return summary 