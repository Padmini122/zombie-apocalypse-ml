
import numpy as np
import pandas as pd
import networkx as nx

class ZombieApocalypseSimulator:
    def __init__(self, num_cities=20, map_seed=42):
        np.random.seed(map_seed)
        self.num_cities = num_cities
        self.G = None
        self.city_data = None
        self.outbreak_history = None
        
    def _generate_city_names(self):
        """Generate unique city names for any number of cities"""
        prefixes = ["North", "South", "East", "West", "New", "Old", "Fort", "Port", "Mount", "Lake"]
        suffixes = ["ville", "town", "burg", "ford", "field", "gate", "haven", "crest", "wood", "shire"]
        
        names = []
        for i in range(self.num_cities):
            prefix = prefixes[i % len(prefixes)]
            suffix = suffixes[(i * 7) % len(suffixes)]
            names.append(f"{prefix}{suffix}")
        return names
        
    def create_city_network(self):
        """Create city network that works with any num_cities"""
        # Adjust connection radius based on city count
        radius = max(0.15, min(0.5, 3.0 / self.num_cities))
        self.G = nx.random_geometric_graph(self.num_cities, radius, seed=42)
        
        # Ensure graph is connected
        while not nx.is_connected(self.G):
            components = list(nx.connected_components(self.G))
            if len(components) > 1:
                u = list(components[0])[0]
                v = list(components[1])[0]
                self.G.add_edge(u, v)
        
        city_names = self._generate_city_names()
        
        # Generate all arrays with exact same length
        n = self.num_cities
        self.city_data = pd.DataFrame({
            'city_name': city_names,
            'population': np.random.randint(5000, 500000, n),
            'hospital_count': np.random.randint(0, 5, n),
            'police_stations': np.random.randint(0, 3, n),
            'food_supplies': np.random.randint(100, 10000, n),
            'weapon_stores': np.random.randint(0, 4, n),
            'has_walls': np.random.choice([0, 1], n, p=[0.7, 0.3]),
            'elevation': np.random.randint(0, 1000, n),
        })
        
        pos = nx.get_node_attributes(self.G, 'pos')
        for i in range(n):
            self.G.nodes[i]['name'] = city_names[i]
            self.G.nodes[i]['pos'] = pos[i]
            self.G.nodes[i]['status'] = 'safe'
            self.G.nodes[i]['population'] = int(self.city_data.iloc[i]['population'])
            
        return self.city_data
    
    def simulate_outbreak(self, start_city=0, days=30, infection_rate=0.3):
        """Simulate outbreak with proper bounds checking"""
        if start_city >= self.num_cities:
            start_city = 0
            
        infection_prob = np.zeros(self.num_cities)
        infection_prob[start_city] = 1.0
        outbreak_log = []
        
        for day in range(days):
            new_infection_prob = infection_prob.copy()
            
            for node in range(self.num_cities):
                if infection_prob[node] > 0.5:
                    for neighbor in list(self.G.neighbors(node)):
                        pos1 = np.array(self.G.nodes[node]['pos'])
                        pos2 = np.array(self.G.nodes[neighbor]['pos'])
                        distance = np.linalg.norm(pos1 - pos2)
                        distance = max(distance, 0.01)  # Prevent division by zero
                        
                        spread_chance = infection_rate * (1 / (1 + distance))
                        
                        pop = self.city_data.iloc[neighbor]['population']
                        pop_factor = max(0.1, pop / 500000)
                        spread_chance *= (1 + pop_factor)
                        
                        elev = self.city_data.iloc[neighbor]['elevation']
                        elev_factor = max(0.1, 1 - (elev / 2000))
                        spread_chance *= elev_factor
                        
                        new_infection_prob[neighbor] = min(1.0, 
                            new_infection_prob[neighbor] + spread_chance)
            
            infection_prob = np.clip(new_infection_prob, 0, 1)
            
            for node in range(self.num_cities):
                if infection_prob[node] > 0.7:
                    status = 2
                elif infection_prob[node] > 0.3:
                    status = 1
                else:
                    status = 0
                
                outbreak_log.append({
                    'day': day,
                    'city_id': node,
                    'city_name': self.city_data.iloc[node]['city_name'],
                    'status': status,
                    'infection_probability': float(infection_prob[node])
                })
        
        self.outbreak_history = pd.DataFrame(outbreak_log)
        return self.outbreak_history

print("✅ Simulator module loaded!")