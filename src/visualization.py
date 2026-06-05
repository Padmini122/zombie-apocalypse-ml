
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
import os

plt.style.use('dark_background')

class OutbreakVisualizer:
    def __init__(self, simulator, ml_data, predictor, feature_engineer):
        self.simulator = simulator
        self.ml_data = ml_data
        self.predictor = predictor
        self.feature_engineer = feature_engineer
        os.makedirs('visualizations', exist_ok=True)
        
    def plot_outbreak_progression(self):
        print("   Creating outbreak progression map...")
        days_total = len(self.simulator.outbreak_history['day'].unique())
        step = max(1, days_total // 6)
        days = [0, step, step*2, step*3, step*4, days_total-1]
        days = [min(d, days_total-1) for d in days]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, day in enumerate(days):
            pos = nx.get_node_attributes(self.simulator.G, 'pos')
            day_data = self.simulator.outbreak_history[self.simulator.outbreak_history['day'] == day]
            
            if len(day_data) == 0:
                continue
                
            node_colors = []
            for node in range(self.simulator.num_cities):
                node_rows = day_data[day_data['city_id'] == node]
                if len(node_rows) > 0:
                    status = node_rows['status'].values[0]
                else:
                    status = 0
                    
                if status == 2:
                    node_colors.append('#FF0000')
                elif status == 1:
                    node_colors.append('#FFA500')
                else:
                    node_colors.append('#00FF00')
            
            nx.draw_networkx_edges(self.simulator.G, pos, alpha=0.2, edge_color='white', ax=axes[idx])
            nx.draw_networkx_nodes(self.simulator.G, pos, node_color=node_colors, node_size=300, alpha=0.8, ax=axes[idx])
            axes[idx].set_title(f'Day {day}', fontsize=14, fontweight='bold')
            axes[idx].axis('off')
        
        plt.suptitle('Zombie Apocalypse Progression', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig('visualizations/outbreak_progression.png', dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        print(f"   ✅ Saved: visualizations/outbreak_progression.png")
    
    def plot_confusion_matrices(self, results):
        print("   Creating confusion matrices...")
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, model_name in enumerate(['Linear Regression', 'Decision Tree', 'Random Forest']):
            cm = results[model_name]['confusion_matrix']
            accuracy = results[model_name]['accuracy']
            
            if cm.shape == (2, 2):
                labels = ['Safe', 'Fails']
            else:
                labels = ['Class 0', 'Class 1'] if cm.shape[0] == 2 else ['Single']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                       xticklabels=labels[:cm.shape[1]], 
                       yticklabels=labels[:cm.shape[0]],
                       ax=axes[idx], cbar=False, annot_kws={'size': 16, 'weight': 'bold'})
            axes[idx].set_title(f'{model_name}\nAccuracy: {accuracy:.1%}', fontsize=12, fontweight='bold')
        
        plt.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig('visualizations/confusion_matrices.png', dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        print(f"   ✅ Saved: visualizations/confusion_matrices.png")
    
    def plot_roc_curves(self, results):
        print("   Creating ROC curves...")
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = {'Linear Regression': '#3498db', 'Decision Tree': '#e67e22', 'Random Forest': '#e74c3c'}
        
        for model_name in ['Linear Regression', 'Decision Tree', 'Random Forest']:
            fpr = results[model_name]['fpr']
            tpr = results[model_name]['tpr']
            roc_auc = results[model_name]['auc']
            ax.plot(fpr, tpr, color=colors[model_name], linewidth=3,
                   label=f"{model_name} (AUC = {roc_auc:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC = 0.500)')
        ax.set_xlabel('False Positive Rate', fontsize=13)
        ax.set_ylabel('True Positive Rate', fontsize=13)
        ax.set_title('ROC Curves: Evacuation Strategy', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig('visualizations/roc_curves.png', dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        print(f"   ✅ Saved: visualizations/roc_curves.png")
    
    def plot_feature_importance(self, results):
        print("   Creating feature importance chart...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        fi = results['feature_importance']
        
        for idx, (model_type, color, title) in enumerate([
            ('decision_tree', '#e67e22', 'Decision Tree'),
            ('random_forest', '#e74c3c', 'Random Forest')
        ]):
            importance = fi[model_type]
            features = self.feature_engineer.feature_columns
            
            if len(importance) != len(features):
                features = [f'Feature_{i}' for i in range(len(importance))]
            
            df = pd.DataFrame({
                'feature': features,
                'importance': importance
            }).sort_values('importance', ascending=False).head(10)
            
            axes[idx].barh(df['feature'], df['importance'], color=color, edgecolor='white')
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{title}: Top Features', fontweight='bold')
            axes[idx].invert_yaxis()
        
        plt.suptitle('What Predicts Route Failure?', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig.savefig('visualizations/feature_importance.png', dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        print(f"   ✅ Saved: visualizations/feature_importance.png")

print("✅ Visualization module loaded!")
