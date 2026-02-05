import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import xgboost as xgb

# --- 全局设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False 
warnings.filterwarnings('ignore')

# --- ASWBoost 类 (保持不变) ---
class ASWBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, M=50, theta=0.0, init_balanced=True):
        self.base_estimator = base_estimator
        self.M = M
        self.theta = theta
        self.init_balanced = init_balanced
        
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'])
        self.classes_ = unique_labels(y)
        y_encoded = np.where(y == self.classes_[0], -1, 1)
        
        n_samples = X.shape[0]
        
        # --- 代价敏感初始化 ---
        if self.init_balanced:
            n_pos = np.sum(y_encoded == 1)
            n_neg = np.sum(y_encoded == -1)
            w_pos = 0.5 / n_pos
            w_neg = 0.5 / n_neg
            sample_weights = np.where(y_encoded == 1, w_pos, w_neg)
        else:
            sample_weights = np.full(n_samples, 1/n_samples)
        
        self.estimators_ = []
        self.alphas_ = []
        
        if self.base_estimator is None:
            base_estimator = DecisionTreeClassifier(max_depth=1, splitter='best')
        else:
            base_estimator = self.base_estimator

        for m in range(self.M):
            estimator = clone(base_estimator)
            estimator.fit(X, y_encoded, sample_weight=sample_weights)
            y_pred = estimator.predict(X)
            
            incorrect = (y_pred != y_encoded)
            err = np.average(incorrect, weights=sample_weights, axis=0)
            err = np.clip(err, 1e-10, 1 - 1e-10)
            
            term1 = (1 - err) / ((1 + self.theta) * err)
            if term1 <= 0: term1 = 1e-10
            
            alpha = (1/(2 + self.theta)) * np.log(term1)
            
            margin = y_encoded * y_pred
            adjustment = 1 + self.theta * (1 - margin) / 2
            
            sample_weights *= np.exp(-alpha * margin * adjustment)
            sample_weights /= np.sum(sample_weights)
            
            self.estimators_.append(estimator)
            self.alphas_.append(alpha)
            
            if err > 0.5 or err < 1e-10:
                break
        
        return self
    
    def predict(self, X):
        check_is_fitted(self, ['estimators_', 'alphas_'])
        X = check_array(X)
        all_preds = np.array([est.predict(X) for est in self.estimators_])
        scores = np.dot(self.alphas_, all_preds)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])
    
    def predict_proba(self, X):
        check_is_fitted(self, ['estimators_', 'alphas_'])
        X = check_array(X)
        all_preds = np.array([est.predict(X) for est in self.estimators_])
        scores = np.dot(self.alphas_, all_preds)
        probs = 1 / (1 + np.exp(-2 * scores)) 
        return np.vstack([1 - probs, probs]).T

# --- Evaluate 函数 (保持不变) ---
def evaluate(y_true, y_pred, y_prob, model_name, fold_idx):
    labels = unique_labels(y_true)
    pos_label = labels[-1]
    neg_label = labels[0]
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    g_mean = np.sqrt(sensitivity * specificity)

    return {
        "Model": model_name,
        "Fold": fold_idx,
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, average='macro'),
        "G-mean": g_mean,
        "AUC": roc_auc_score(y_true, y_prob)
    }

def run_scientific_experiment():
    # --- 1. 数据生成: 8:2  ---
    print(">>> 1. 数据生成: 高度不平衡(7:3)")
    X, y = make_classification(
        n_samples=1000, 
        n_features=40, 
        n_informative=25,
        n_redundant=10,
        n_classes=2, 
        weights=[0.9, 0.1],        
        class_sep=1.5,            
        flip_y=0.10,               
        random_state=42
    )
    y = np.where(y == 0, -1, 1) 
    
    N_EST = 50
    
    # === 修复说明：删除了 AdaBoostClassifier 中的 algorithm='SAMME' 参数 ===
    models_config = [
        ("AdaBoost", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1), 
            n_estimators=N_EST, random_state=42)),  # <--- 修改处
            
        ("CS-AdaBoost", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, class_weight='balanced'), 
            n_estimators=N_EST, random_state=42)), 

        ("RobustBoost", GradientBoostingClassifier( # 用较小的 subsample 模拟抗噪
            n_estimators=N_EST, max_depth=3, learning_rate=0.5, subsample=0.3, random_state=42)),

        ("GBDT", GradientBoostingClassifier(
            n_estimators=N_EST, max_depth=3, learning_rate=0.1, random_state=42)),
            
        ("XGBoost", xgb.XGBClassifier(
            n_estimators=N_EST, max_depth=6, learning_rate=0.3, use_label_encoder=False, 
            eval_metric='logloss', n_jobs=1, random_state=42))
    ]

    asw_param_grid = {'theta': [ 0.05, 0.1,0.2,0.3,0.4, 0.5,0.6,0.7, 0.8,0.9, 1.0]} 
    
    final_results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"\n>>> 2. 开始 5-Fold CV...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_01 = np.where(y_train == -1, 0, 1)
        
        print(f"  > Fold {fold_idx + 1}/5 processing...")

        # --- A. ASWBoost (带 Theta 打印) ---
        grid = GridSearchCV(
            ASWBoost(base_estimator=DecisionTreeClassifier(max_depth=1), M=N_EST, init_balanced=True),
            asw_param_grid, cv=3, scoring='balanced_accuracy', n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_asw = grid.best_estimator_
        
        print(f"    [ASWBoost Opt] Best Theta selected: {best_asw.theta}")
        
        final_results.append(evaluate(
            y_test, 
            best_asw.predict(X_test), 
            best_asw.predict_proba(X_test)[:, 1], 
            "ASWBoost", fold_idx
        ))

        # --- B. 其他模型 ---
        for name, model in models_config:
            clf = clone(model)
            if name in ["GBDT", "XGBoost"]:
                clf.fit(X_train, y_train_01)
                y_pred_raw = clf.predict(X_test)
                y_pred = np.where(y_pred_raw == 0, -1, 1)
                y_prob = clf.predict_proba(X_test)[:, 1]
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1]
            
            final_results.append(evaluate(y_test, y_pred, y_prob, name, fold_idx))

    # --- 结果汇总 ---
    df = pd.DataFrame(final_results)
    df_avg = df.groupby("Model")[["Balanced Acc", "G-mean", "Macro-F1", "AUC"]].mean()
    df_std = df.groupby("Model")[["Balanced Acc"]].std()
    
    df_avg = df_avg.sort_values("G-mean", ascending=False)
    
    print("\n" + "="*80)
    print("实验最终结果 (5-Fold Average) - 7:3 高不平衡 + 噪声")
    print("="*80)
    print(df_avg.round(4))
    print("\n标准差 (稳定性):")
    print(df_std.round(4))

if __name__ == "__main__":
    run_scientific_experiment()
    