
import sys
sys.path.append('src')

import numpy as np
import os

# Create output folders
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

from simulator import ZombieApocalypseSimulator
from feature_engineering import RouteFeatureEngineer
from models import RouteFailurePredictor
from visualization import OutbreakVisualizer
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("🧟 ZOMBIE APOCALYPSE SUPPLY CHAIN OPTIMIZER")
print("="*60 + "\n")

# Step 1: Create the world
print("🌍 Creating virtual world...")
simulator = ZombieApocalypseSimulator(num_cities=20, map_seed=42)
city_data = simulator.create_city_network()
print(f"✅ Created {len(city_data)} cities")

# Step 2: Start the outbreak
print("\n🧟 Simulating zombie outbreak...")
outbreak_data = simulator.simulate_outbreak(start_city=0, days=30)
print(f"✅ Simulated {len(outbreak_data)} records")

# Step 3: Engineer features
print("\n🔧 Engineering features for ML...")
engineer = RouteFeatureEngineer(outbreak_data, city_data, simulator.G)
ml_data = engineer.create_features()
print(f"✅ Created {len(ml_data)} samples with {len(engineer.feature_columns)} features")

# Step 4: Prepare train/test split
print("\n📊 Preparing training and test sets...")
X_train, X_test, y_train, y_test, scaler = engineer.prepare_train_test_split(ml_data)
print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")

# Step 5: Train models
print("\n🤖 Training models...")
predictor = RouteFailurePredictor()
predictor.train_all_models(X_train, y_train)

# Step 6: Evaluate
print("\n📈 Evaluating models...")
results = predictor.evaluate_all(X_test, y_test)

for model_name in ['Linear Regression', 'Decision Tree', 'Random Forest']:
    r = results[model_name]
    print(f"\n{model_name}:")
    print(f"  Accuracy: {r['accuracy']:.2%}")
    print(f"  AUC: {r['auc']:.3f}")

# Step 7: Visualizations
print("\n🎨 Creating visualizations...")
results['feature_importance']['feature_names'] = engineer.feature_columns
visualizer = OutbreakVisualizer(simulator, ml_data, predictor, engineer)
visualizer.plot_outbreak_progression()
visualizer.plot_confusion_matrices(results)
visualizer.plot_roc_curves(results)
visualizer.plot_feature_importance(results)

# Step 8: Save models
print("\n💾 Saving models...")
predictor.save_models()

print("\n" + "="*60)
print("✅ PROJECT COMPLETE!")
print("="*60)
print("\n📁 Files created:")
print("   models/linear_regression.pkl")
print("   models/decision_tree.pkl")
print("   models/random_forest.pkl")
print("   visualizations/outbreak_progression.png")
print("   visualizations/confusion_matrices.png")
print("   visualizations/roc_curves.png")
print("   visualizations/feature_importance.png")
