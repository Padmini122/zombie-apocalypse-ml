
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import joblib

class RouteFailurePredictor:
    def __init__(self):
        self.lr_model = None
        self.dt_model = None
        self.rf_model = None
        
    def train_all_models(self, X_train, y_train):
        print("   Training Linear Regression...")
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train, y_train)
        
        print("   Training Decision Tree...")
        self.dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=max(2, len(y_train)//10), 
                                                min_samples_leaf=max(1, len(y_train)//20), random_state=42)
        self.dt_model.fit(X_train, y_train)
        
        print("   Training Random Forest...")
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                                min_samples_split=max(2, len(y_train)//20), 
                                                min_samples_leaf=max(1, len(y_train)//40),
                                                class_weight='balanced', random_state=42)
        self.rf_model.fit(X_train, y_train)
        print("   ✅ All models trained!")
        
    def calculate_metrics(self, y_true, y_pred):
        """Calculate metrics safely for any class distribution"""
        unique_classes = np.unique(y_true)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        if len(unique_classes) > 1:
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        else:
            only_class = unique_classes[0]
            metrics['precision'] = 1.0 if only_class == 1 else 0.0
            metrics['recall'] = 1.0 if only_class == 1 else 0.0
            metrics['f1'] = 1.0 if only_class == 1 else 0.0
            
        return metrics
        
    def evaluate_all(self, X_test, y_test):
        """Evaluate all models with robust error handling"""
        results = {}
        
        # Linear Regression
        lr_pred_raw = self.lr_model.predict(X_test)
        lr_pred = np.clip(lr_pred_raw, 0, 1)
        lr_binary = (lr_pred > 0.5).astype(int)
        
        # Decision Tree
        dt_pred = self.dt_model.predict(X_test)
        
        # Random Forest
        rf_pred = self.rf_model.predict(X_test)
        
        # Get probabilities safely
        try:
            dt_proba = self.dt_model.predict_proba(X_test)
            if dt_proba.shape[1] > 1:
                dt_proba = dt_proba[:, 1]
            else:
                dt_proba = dt_proba[:, 0]
        except:
            dt_proba = dt_pred.astype(float)
            
        try:
            rf_proba = self.rf_model.predict_proba(X_test)
            if rf_proba.shape[1] > 1:
                rf_proba = rf_proba[:, 1]
            else:
                rf_proba = rf_proba[:, 0]
        except:
            rf_proba = rf_pred.astype(float)
        
        # Evaluate each model
        for name, preds, probas in [
            ('Linear Regression', lr_binary, lr_pred),
            ('Decision Tree', dt_pred, dt_proba),
            ('Random Forest', rf_pred, rf_proba)
        ]:
            metrics = self.calculate_metrics(y_test, preds)
            
            # ROC curve
            unique_test = np.unique(y_test)
            if len(unique_test) > 1:
                try:
                    fpr, tpr, _ = roc_curve(y_test, probas)
                    roc_auc = auc(fpr, tpr)
                except:
                    fpr = np.array([0.0, 1.0])
                    tpr = np.array([0.0, 1.0])
                    roc_auc = 0.5
            else:
                fpr = np.array([0.0, 1.0])
                tpr = np.array([0.0, 1.0])
                roc_auc = 0.5
            
            results[name] = {
                'predictions': preds,
                'probabilities': probas,
                'confusion_matrix': metrics['confusion_matrix'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc
            }
        
        # Feature importance
        results['feature_importance'] = {
            'decision_tree': self.dt_model.feature_importances_,
            'random_forest': self.rf_model.feature_importances_
        }
        
        return results
    
    def save_models(self):
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.lr_model, 'models/linear_regression.pkl')
        joblib.dump(self.dt_model, 'models/decision_tree.pkl')
        joblib.dump(self.rf_model, 'models/random_forest.pkl')
        print("   ✅ Models saved!")

print("✅ Models module loaded!")
