import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.datasets import fetch_openml
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, balanced_accuracy_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

# 引入 imblearn 用于 RUSBoost 和 SMOTE
try:
    from imblearn.ensemble import RUSBoostClassifier
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError("请安装 imbalanced-learn 库: pip install imbalanced-learn")

# --- 全局设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False 
warnings.filterwarnings('ignore')

# --- SMOTEBoost 手动实现类 ---
class SMOTEBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=50, k_neighbors=5, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        n_samples, _ = X.shape
        
        # 标签编码为 0, 1
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # 初始化权重
        sample_weights = np.full(n_samples, 1 / n_samples)
        
        self.estimators_ = []
        self.alphas_ = []
        self.le_ = le
        
        base_estimator = self.base_estimator
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(max_depth=1)
            
        rng = np.random.RandomState(self.random_state)
        
        for m in range(self.n_estimators):
            # 1. SMOTE 生成新数据 (仅针对少数类)
            # 为了保证多样性，每次SMOTE可以基于当前分布或直接对原数据SMOTE
            # 标准SMOTEBoost是在每一轮对数据进行SMOTE，然后训练
            try:
                sm = SMOTE(k_neighbors=self.k_neighbors, random_state=rng)
                X_res, y_res = sm.fit_resample(X, y_encoded)
            except ValueError:
                # 如果样本太少无法SMOTE，回退到原数据
                X_res, y_res = X, y_encoded
            
            # 2. 训练弱分类器 (注意：SMOTE后的数据权重通常重置或均匀，
            # 但为了结合Boosting，通常做法是训练时不带权重或带均匀权重，
            # 然后在原数据集上计算误差更新权重。这里采用在Resampled数据上fit)
            estimator = clone(base_estimator)
            estimator.fit(X_res, y_res)
            
            # 3. 在原始数据集上计算误差
            y_pred = estimator.predict(X)
            incorrect = (y_pred != y_encoded)
            
            # 考虑样本权重计算误差
            err = np.average(incorrect, weights=sample_weights, axis=0)
            
            # 4. 计算 Alpha 和 更新权重
            if err > 0.5:
                # 误差过大，停止或反转，这里简单break
                break
            if err <= 1e-10:
                err = 1e-10
                
            alpha = 0.5 * np.log((1 - err) / err)
            
            # 更新权重: w * exp(-alpha * y * h(x)) 
            # 等价于: 正确样本 * exp(-alpha), 错误样本 * exp(alpha)
            # 为了数值稳定，通常使用 indicator
            
            # 错误样本权重增加
            sample_weights *= np.exp(alpha * (incorrect * 2 - 1)) # incorrect=1 -> exp(alpha), incorrect=0 -> exp(-alpha)
            sample_weights /= np.sum(sample_weights)
            
            self.estimators_.append(estimator)
            self.alphas_.append(alpha)
            
        return self

    def predict(self, X):
        check_is_fitted(self, ['estimators_', 'alphas_'])
        X = check_array(X)
        
        # 累加加权投票
        pred_scores = np.zeros((X.shape[0], len(self.classes_)))
        
        for estimator, alpha in zip(self.estimators_, self.alphas_):
            # 简单二分类处理
            preds = estimator.predict(X)
            # preds是 0 或 1
            # 将 0 映射为 -1 进行计算，或直接加分
            for i, p in enumerate(preds):
                if p == 1:
                    pred_scores[i, 1] += alpha
                else:
                    pred_scores[i, 0] += alpha
                    
        final_preds = np.argmax(pred_scores, axis=1)
        return self.le_.inverse_transform(final_preds)
    
    def predict_proba(self, X):
        # 简化的概率输出
        check_is_fitted(self, ['estimators_', 'alphas_'])
        X = check_array(X)
        pred_scores = np.zeros((X.shape[0], len(self.classes_)))
        
        sum_alphas = sum(self.alphas_) if sum(self.alphas_) > 0 else 1
        
        for estimator, alpha in zip(self.estimators_, self.alphas_):
            # 获取基分类器的概率
            try:
                probas = estimator.predict_proba(X)
                pred_scores += probas * alpha
            except:
                # 如果基分类器没有proba，使用硬投票
                preds = estimator.predict(X)
                for i, p in enumerate(preds):
                    pred_scores[i, p] += alpha
                    
        pred_scores /= sum_alphas
        return pred_scores

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
        # 确保二分类标签为 -1, 1
        y_encoded = np.where(y == self.classes_[0], -1, 1)
        
        n_samples = X.shape[0]
        
        if self.init_balanced:
            n_pos = np.sum(y_encoded == 1)
            n_neg = np.sum(y_encoded == -1)
            w_pos = 0.5 / n_pos if n_pos > 0 else 0
            w_neg = 0.5 / n_neg if n_neg > 0 else 0
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

# --- Evaluate 函数 (增加 Recall) ---
def evaluate(y_true, y_pred, y_prob, model_name, fold_idx):
    labels = unique_labels(y_true)
    # 假设少数类/正类是最后一个标签 (1)
    pos_label = labels[-1]
    neg_label = labels[0]
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    g_mean = np.sqrt(sensitivity * specificity)
    
    # Recall Score
    rec = recall_score(y_true, y_pred, pos_label=pos_label)

    return {
        "Model": model_name,
        "Fold": fold_idx,
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "Macro-F1": f1_score(y_true, y_pred, average='macro'),
        "G-mean": g_mean,
        "Recall": rec,
        "AUC": roc_auc_score(y_true, y_prob) if y_prob is not None else 0
    }

def get_real_medical_data():
    """
    获取 Blood Transfusion Service Center 数据集 (OpenML ID: 1464)
    这是一个真实的、不平衡的医学/生物数据集。
    """
    print(">>> 正在加载数据集: Blood Transfusion Service Center (OpenML ID: 1464)...")
    try:
        # 使用 fetch_openml 获取数据
        data = fetch_openml(data_id=1464, as_frame=True, parser='auto')
        X = data.data
        y = data.target
        
        # 预处理
        # 目标变量通常是 '1', '2' 或 '0', '1'，我们将其转换为 0 (多数类/负类) 和 1 (少数类/正类)
        le = LabelEncoder()
        y = le.fit_transform(y) # 这里的 '2' (donated) 通常是少数类
        
        # 确保 1 是少数类
        if np.sum(y == 1) > np.sum(y == 0):
            y = 1 - y
            
        # 简单标准化特征
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        print(f"    数据加载完成. 样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
        print(f"    类别分布: 0 (负类): {np.sum(y==0)}, 1 (正类): {np.sum(y==1)}")
        print(f"    不平衡比例: 1 : {np.sum(y==0)/np.sum(y==1):.2f}")
        
        return X, y
        
    except Exception as e:
        print(f"    在线加载失败 ({e})，尝试使用模拟数据代替以保证代码运行...")
        from sklearn.datasets import make_classification
        return make_classification(n_samples=748, n_features=4, weights=[0.76, 0.24], random_state=42)

def run_scientific_experiment():
    # --- 1. 数据获取 ---
    X, y = get_real_medical_data()
    # ASWBoost 需要 y 为 -1, 1 格式进行内部计算，但 gridsearchcv 会处理
    # 为了统一，我们在传入模型前保持 0, 1，模型内部处理
    
    N_EST = 50 # 树的数量
    RANDOM_STATE = 42
    
    # --- 2. 模型配置 (7个模型) ---
    models_config = [
        # 1. Standard AdaBoost
        ("AdaBoost", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1), 
            n_estimators=N_EST, algorithm='SAMME', random_state=RANDOM_STATE)),
            
        # 2. Cost-Sensitive AdaBoost (通过class_weight实现)
        ("CS-AdaBoost", AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, class_weight='balanced'), 
            n_estimators=N_EST, algorithm='SAMME', random_state=RANDOM_STATE)),

        # 3. RUSBoost (Standard from imblearn)
        ("RUSBoost", RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=N_EST, sampling_strategy='auto', random_state=RANDOM_STATE)),

        # 4. SMOTEBoost (Custom Implementation)
        ("SMOTEBoost", SMOTEBoost(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=N_EST, k_neighbors=5, random_state=RANDOM_STATE)),

        # 5. GBDT
        ("GBDT", GradientBoostingClassifier(
            n_estimators=N_EST, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE)),
            
        # 6. XGBoost
        ("XGBoost", xgb.XGBClassifier(
            n_estimators=N_EST, max_depth=6, learning_rate=0.1, use_label_encoder=False, 
            eval_metric='logloss', n_jobs=1, random_state=RANDOM_STATE))
    ]

    # ASWBoost 参数网格 (精细化 -1 到 1)
    asw_param_grid = {'theta': np.linspace(-0.99, 0.99, 20)} 
    
    final_results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    print(f"\n>>> 2. 开始 5-Fold CV 对比实验...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"  > Fold {fold_idx + 1}/5 processing...")

        # --- A. ASWBoost (Grid Search & Best Theta) ---
        # 注意: ASWBoost 内部会将标签转为 -1, 1，这里传入 0, 1 即可
        grid = GridSearchCV(
            ASWBoost(base_estimator=DecisionTreeClassifier(max_depth=1), M=N_EST, init_balanced=True),
            asw_param_grid, cv=3, scoring='balanced_accuracy', n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_asw = grid.best_estimator_
        
        print(f"    [ASWBoost] Best Theta: {best_asw.theta:.4f}")
        
        final_results.append(evaluate(
            y_test, 
            best_asw.predict(X_test), 
            best_asw.predict_proba(X_test)[:, 1], 
            "ASWBoost", fold_idx
        ))

        # --- B. 其他 6 个模型 ---
        for name, model in models_config:
            clf = clone(model)
            # XGBoost 和 GBDT 等通常不需要特殊转换标签，0/1即可
            try:
                clf.fit(X_train, y_train)
            except Exception as e:
                print(f"    Error fitting {name}: {e}")
                continue
                
            y_pred = clf.predict(X_test)
            try:
                y_prob = clf.predict_proba(X_test)[:, 1]
            except:
                y_prob = None # SMOTEBoost 如果没有概率输出
            
            final_results.append(evaluate(y_test, y_pred, y_prob, name, fold_idx))

    # --- 结果汇总 ---
    df = pd.DataFrame(final_results)
    # 增加 Recall 到汇总
    metrics = ["Balanced Acc", "G-mean", "Macro-F1", "Recall", "AUC"]
    df_avg = df.groupby("Model")[metrics].mean()
    df_std = df.groupby("Model")[["Balanced Acc"]].std()
    
    df_avg = df_avg.sort_values("Balanced Acc", ascending=False)
    
    print("\n" + "="*100)
    print("实验最终结果 (5-Fold Average) - Dataset: Blood Transfusion Service Center")
    print("="*100)
    print(df_avg.round(4))
    print("\n标准差 (稳定性 - Balanced Acc):")
    print(df_std.round(4))
    print("="*100)

if __name__ == "__main__":
    run_scientific_experiment()