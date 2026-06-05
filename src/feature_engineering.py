
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class RouteFeatureEngineer:
    def __init__(self, outbreak_data, city_data, graph):
        self.outbreak_data = outbreak_data
        self.city_data = city_data
        self.graph = graph
        self.feature_columns = None
        
    def create_features(self, start_day=0, end_day=25):
        """Create features with bounds checking"""
        max_day = self.outbreak_data['day'].max()
        end_day = min(end_day, max_day)
        
        features_list = []
        
        for day in range(start_day, end_day):
            day_data = self.outbreak_data[self.outbreak_data['day'] == day]
            
            if len(day_data) == 0:
                continue
                
            for city_id in range(len(self.city_data)):
                neighbors = list(self.graph.neighbors(city_id))
                
                for neighbor_id in neighbors:
                    city_info = self.city_data.iloc[city_id]
                    neighbor_info = self.city_data.iloc[neighbor_id]
                    
                    pos1 = np.array(self.graph.nodes[city_id]['pos'])
                    pos2 = np.array(self.graph.nodes[neighbor_id]['pos'])
                    distance = max(np.linalg.norm(pos1 - pos2), 0.01)
                    
                    # Safe data extraction
                    city_row = day_data[day_data['city_id'] == city_id]
                    neigh_row = day_data[day_data['city_id'] == neighbor_id]
                    
                    if len(city_row) == 0 or len(neigh_row) == 0:
                        continue
                    
                    city_inf = float(city_row['infection_probability'].values[0])
                    neigh_inf = float(neigh_row['infection_probability'].values[0])
                    
                    infected_neighbors = 0
                    for n in neighbors:
                        n_row = day_data[day_data['city_id'] == n]
                        if len(n_row) > 0:
                            if n_row['status'].values[0] >= 1:
                                infected_neighbors += 1
                    
                    # Target: Will route fail tomorrow?
                    route_will_fail = 0
                    if day + 1 <= max_day:
                        tomorrow = self.outbreak_data[self.outbreak_data['day'] == day + 1]
                        c_tom = tomorrow[tomorrow['city_id'] == city_id]
                        n_tom = tomorrow[tomorrow['city_id'] == neighbor_id]
                        
                        if len(c_tom) > 0 and len(n_tom) > 0:
                            if c_tom['status'].values[0] >= 2 or n_tom['status'].values[0] >= 2:
                                route_will_fail = 1
                    
                    features_list.append({
                        'day': day,
                        'city_id': city_id,
                        'neighbor_id': neighbor_id,
                        'city_name': str(city_info['city_name']),
                        'neighbor_name': str(neighbor_info['city_name']),
                        'distance': distance,
                        'city_population': int(city_info['population']),
                        'neighbor_population': int(neighbor_info['population']),
                        'city_elevation': int(city_info['elevation']),
                        'neighbor_elevation': int(neighbor_info['elevation']),
                        'city_has_walls': int(city_info['has_walls']),
                        'neighbor_has_walls': int(neighbor_info['has_walls']),
                        'city_food_supplies': int(city_info['food_supplies']),
                        'neighbor_food_supplies': int(neighbor_info['food_supplies']),
                        'city_weapon_stores': int(city_info['weapon_stores']),
                        'neighbor_weapon_stores': int(neighbor_info['weapon_stores']),
                        'city_hospitals': int(city_info['hospital_count']),
                        'neighbor_hospitals': int(neighbor_info['hospital_count']),
                        'city_infection_prob': city_inf,
                        'neighbor_infection_prob': neigh_inf,
                        'infected_neighbors_count': infected_neighbors,
                        'avg_infection_prob_region': (city_inf + neigh_inf) / 2,
                        'route_will_fail': route_will_fail
                    })
        
        if len(features_list) == 0:
            # Return empty dataframe with correct columns
            self.feature_columns = [
                'distance', 'city_population', 'neighbor_population',
                'city_elevation', 'neighbor_elevation', 'city_has_walls',
                'neighbor_has_walls', 'city_food_supplies', 'neighbor_food_supplies',
                'city_weapon_stores', 'neighbor_weapon_stores',
                'city_hospitals', 'neighbor_hospitals',
                'city_infection_prob', 'neighbor_infection_prob',
                'infected_neighbors_count', 'avg_infection_prob_region'
            ]
            return pd.DataFrame(columns=self.feature_columns + ['day', 'city_id', 'neighbor_id', 
                                   'city_name', 'neighbor_name', 'route_will_fail'])
        
        features_df = pd.DataFrame(features_list)
        
        self.feature_columns = [
            'distance', 'city_population', 'neighbor_population',
            'city_elevation', 'neighbor_elevation', 'city_has_walls',
            'neighbor_has_walls', 'city_food_supplies', 'neighbor_food_supplies',
            'city_weapon_stores', 'neighbor_weapon_stores',
            'city_hospitals', 'neighbor_hospitals',
            'city_infection_prob', 'neighbor_infection_prob',
            'infected_neighbors_count', 'avg_infection_prob_region'
        ]
        
        return features_df
    
    def prepare_train_test_split(self, features_df, test_day_threshold=20):
        """Split data safely even with small datasets"""
        if len(features_df) == 0:
            raise ValueError("No features generated. Try increasing simulation days.")
        
        train_mask = features_df['day'] <= test_day_threshold
        test_mask = features_df['day'] > test_day_threshold
        
        # Ensure both splits have data
        if test_mask.sum() == 0:
            # Use last 20% for test if time split fails
            split_idx = int(len(features_df) * 0.8)
            train_mask = pd.Series(False, index=features_df.index)
            test_mask = pd.Series(False, index=features_df.index)
            train_mask.iloc[:split_idx] = True
            test_mask.iloc[split_idx:] = True
        
        X_train = features_df[train_mask][self.feature_columns].values
        X_test = features_df[test_mask][self.feature_columns].values
        y_train = features_df[train_mask]['route_will_fail'].values
        y_test = features_df[test_mask]['route_will_fail'].values
        
        # Handle empty cases
        if len(X_train) == 0 or len(X_test) == 0:
            # Split 50-50
            mid = len(features_df) // 2
            X_train = features_df.iloc[:mid][self.feature_columns].values
            X_test = features_df.iloc[mid:][self.feature_columns].values
            y_train = features_df.iloc[:mid]['route_will_fail'].values
            y_test = features_df.iloc[mid:]['route_will_fail'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

print("✅ Feature engineering module loaded!")
