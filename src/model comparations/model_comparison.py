"""
Script para comparar múltiples modelos de machine learning
y determinar cuál es el mejor para el problema de scoring crediticio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, log_loss, brier_score_loss
)

import joblib
import json

class ModelComparator:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.trained_models = {}
        
    def add_model(self, name, model):
        """Añade un modelo al comparador"""
        self.models[name] = model
        
    def setup_default_models(self):
        """Configura modelos por defecto optimizados para scoring crediticio"""
        
        self.models = {
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),

            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            
        }
    
    def calculate_ks_statistic(self, y_true, y_pred_proba):
        """Calcula KS Statistic específico para scoring crediticio"""
        bad_rate = []
        good_rate = []
        thresholds = np.linspace(0, 1, 100)
        
        for thresh in thresholds:
            pred_thresh = (y_pred_proba >= thresh).astype(int)
            if len(pred_thresh[pred_thresh == 1]) > 0:
                bad_captured = np.sum((y_true == 0) & (pred_thresh == 1)) / max(np.sum(y_true == 0), 1)
                good_captured = np.sum((y_true == 1) & (pred_thresh == 1)) / max(np.sum(y_true == 1), 1)
            else:
                bad_captured = 0
                good_captured = 0
            bad_rate.append(bad_captured)
            good_rate.append(good_captured)
        
        ks_stat = max(np.array(good_rate) - np.array(bad_rate))
        return ks_stat
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """Evalúa un modelo individual"""
        print(f"   Evaluando {model_name}...")
        
        # Medir tiempo de entrenamiento
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Medir tiempo de predicción
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        prediction_time = time.time() - start_time
        
        # Métricas principales
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'average_precision': average_precision_score(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'ks_statistic': self.calculate_ks_statistic(y_test, y_pred_proba),
            'training_time': training_time,
            'prediction_time': prediction_time,
            'prediction_speed': len(y_test) / prediction_time  # predicciones por segundo
        }
        
        # Validación cruzada
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        metrics['cv_roc_auc_mean'] = np.mean(cv_scores)
        metrics['cv_roc_auc_std'] = np.std(cv_scores)
        
        return metrics, model
    
    def compare_models(self, X, y, test_size=0.2):
        """Compara todos los modelos configurados"""
        print("🔄 Iniciando comparación de modelos...")
        
        # Split de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Evaluar cada modelo
        for name, model in self.models.items():
            try:
                metrics, trained_model = self.evaluate_model(
                    model, X_train, X_test, y_train, y_test, name
                )
                self.results[name] = metrics
                self.trained_models[name] = trained_model
            except Exception as e:
                print(f"   ❌ Error con {name}: {e}")
                continue
        
        return self.results
    
    def create_comparison_table(self):
        """Crea una tabla comparativa de resultados"""
        if not self.results:
            print("❌ No hay resultados para comparar")
            return None
            
        df_results = pd.DataFrame(self.results).T
        
        # Redondear valores para mejor visualización
        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
                       'average_precision', 'log_loss', 'brier_score', 'ks_statistic',
                       'cv_roc_auc_mean', 'cv_roc_auc_std', 'training_time', 
                       'prediction_time', 'prediction_speed']
        
        for col in numeric_cols:
            if col in df_results.columns:
                df_results[col] = pd.to_numeric(df_results[col], errors='coerce').round(4)
        
        # Ordenar por ROC-AUC
        df_results = df_results.sort_values('roc_auc', ascending=False)
        
        return df_results
    
    def plot_comparison(self, save_path="artifacts/model_comparison.png"):
        """Genera gráficos de comparación"""
        if not self.results:
            print("❌ No hay resultados para visualizar")
            return
            
        df_results = self.create_comparison_table()
        
        # Configurar figura
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Métricas a visualizar
        metrics = ['roc_auc', 'precision', 'recall', 'f1_score', 'ks_statistic', 'cv_roc_auc_mean']
        titles = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score', 'KS Statistic', 'CV ROC-AUC']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Crear gráfico de barras
            values = df_results[metric].values
            names = df_results.index
            
            bars = ax.bar(range(len(values)), values)
            ax.set_xlabel('Modelos')
            ax.set_ylabel(title)
            ax.set_title(f'Comparación: {title}')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            
            # Colorear el mejor modelo
            max_idx = np.argmax(values)
            bars[max_idx].set_color('gold')
            
            # Añadir valores en las barras
            for j, v in enumerate(values):
                ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_best_model(self, metric='roc_auc'):
        """Obtiene el mejor modelo según una métrica específica"""
        if not self.results:
            return None
            
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x][metric])
        
        return {
            'name': best_model_name,
            'model': self.trained_models[best_model_name],
            'metrics': self.results[best_model_name]
        }
    
    def save_comparison_report(self, filepath="artifacts/model_comparison_report.json"):
        """Guarda reporte completo de la comparación"""
        if not self.results:
            print("❌ No hay resultados para guardar")
            return
            
        # Añadir ranking por métricas principales
        metrics_ranking = {}
        key_metrics = ['roc_auc', 'precision', 'recall', 'f1_score', 'ks_statistic']
        
        for metric in key_metrics:
            ranking = sorted(self.results.items(), 
                           key=lambda x: x[1][metric], reverse=True)
            metrics_ranking[f'{metric}_ranking'] = [item[0] for item in ranking]
        
        # Mejor modelo general
        best_model = self.get_best_model('roc_auc')
        
        report = {
            'comparison_results': self.results,
            'rankings': metrics_ranking,
            'best_model_overall': {
                'name': best_model['name'],
                'metrics': best_model['metrics']
            },
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': self.generate_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"✅ Reporte guardado en: {filepath}")
        return report
    
    def generate_summary(self):
        """Genera un resumen ejecutivo de la comparación"""
        if not self.results:
            return "No hay resultados disponibles"
            
        best_roc = self.get_best_model('roc_auc')
        best_ks = self.get_best_model('ks_statistic')
        best_precision = self.get_best_model('precision')
        
        summary = {
            'total_models_evaluated': len(self.results),
            'best_overall_model': best_roc['name'],
            'best_roc_auc': f"{best_roc['name']} ({best_roc['metrics']['roc_auc']:.4f})",
            'best_ks_statistic': f"{best_ks['name']} ({best_ks['metrics']['ks_statistic']:.4f})",
            'best_precision': f"{best_precision['name']} ({best_precision['metrics']['precision']:.4f})",
            'recommendations': []
        }
        
        # Generar recomendaciones
        if best_roc['metrics']['roc_auc'] > 0.85:
            summary['recommendations'].append("✅ Excelente discriminación encontrada")
        elif best_roc['metrics']['roc_auc'] < 0.7:
            summary['recommendations'].append("⚠️ Considerar feature engineering adicional")
            
        if best_ks['metrics']['ks_statistic'] > 0.4:
            summary['recommendations'].append("✅ Muy buen poder de separación")
        elif best_ks['metrics']['ks_statistic'] < 0.2:
            summary['recommendations'].append("⚠️ Poder de separación limitado")
            
        return summary
    
    def analyze_performance_comparison(self):
        """Analiza y compara el performance de todos los modelos"""
        if not self.results:
            print("❌ No hay resultados para analizar")
            return None
            
        print("\n📊 ANÁLISIS COMPARATIVO DE PERFORMANCE")
        print("=" * 50)
        
        df_results = self.create_comparison_table()
        
        # Análisis de métricas principales
        print("\n🎯 RANKING POR MÉTRICAS PRINCIPALES:")
        key_metrics = ['roc_auc', 'precision', 'recall', 'f1_score', 'ks_statistic']
        
        for metric in key_metrics:
            print(f"\n   📈 {metric.replace('_', ' ').upper()}:")
            sorted_models = df_results.sort_values(metric, ascending=False)
            for i, (model, row) in enumerate(sorted_models.head(3).iterrows(), 1):
                medal = ['🥇', '🥈', '🥉'][i-1]
                print(f"      {medal} {model}: {row[metric]:.4f}")
        
        # Análisis de performance temporal
        print("\n⏱️ ANÁLISIS DE PERFORMANCE TEMPORAL:")
        print(f"\n   🚀 VELOCIDAD DE ENTRENAMIENTO (más rápido):")
        sorted_by_training = df_results.sort_values('training_time', ascending=True)
        for i, (model, row) in enumerate(sorted_by_training.head(3).iterrows(), 1):
            medal = ['🥇', '🥈', '🥉'][i-1]
            print(f"      {medal} {model}: {row['training_time']:.3f}s")
        
        print(f"\n   ⚡ VELOCIDAD DE PREDICCIÓN (predicciones/segundo):")
        sorted_by_prediction = df_results.sort_values('prediction_speed', ascending=False)
        for i, (model, row) in enumerate(sorted_by_prediction.head(3).iterrows(), 1):
            medal = ['🥇', '🥈', '🥉'][i-1]
            print(f"      {medal} {model}: {row['prediction_speed']:.0f} pred/s")
        
        # Análisis de balance performance/velocidad
        print("\n⚖️ BALANCE PERFORMANCE vs VELOCIDAD:")
        
        # Normalizar métricas para scoring compuesto
        df_normalized = df_results.copy()
        
        # Normalizar ROC-AUC (mayor es mejor)
        df_normalized['roc_auc_norm'] = (df_normalized['roc_auc'] - df_normalized['roc_auc'].min()) / (df_normalized['roc_auc'].max() - df_normalized['roc_auc'].min())
        
        # Normalizar tiempo de entrenamiento (menor es mejor, invertir)
        df_normalized['training_time_norm'] = 1 - (df_normalized['training_time'] - df_normalized['training_time'].min()) / (df_normalized['training_time'].max() - df_normalized['training_time'].min())
        
        # Normalizar velocidad de predicción (mayor es mejor)
        df_normalized['prediction_speed_norm'] = (df_normalized['prediction_speed'] - df_normalized['prediction_speed'].min()) / (df_normalized['prediction_speed'].max() - df_normalized['prediction_speed'].min())
        
        # Score compuesto (60% performance, 20% velocidad entrenamiento, 20% velocidad predicción)
        df_normalized['composite_score'] = (
            0.6 * df_normalized['roc_auc_norm'] + 
            0.2 * df_normalized['training_time_norm'] + 
            0.2 * df_normalized['prediction_speed_norm']
        )
        
        sorted_by_composite = df_normalized.sort_values('composite_score', ascending=False)
        for i, (model, row) in enumerate(sorted_by_composite.head(3).iterrows(), 1):
            medal = ['🥇', '🥈', '🥉'][i-1]
            print(f"      {medal} {model}: Score {row['composite_score']:.3f}")
            print(f"          ROC-AUC: {df_results.loc[model, 'roc_auc']:.4f} | "
                  f"Entrenamiento: {df_results.loc[model, 'training_time']:.2f}s | "
                  f"Predicción: {df_results.loc[model, 'prediction_speed']:.0f} pred/s")
        
        # Análisis de estabilidad
        print("\n📊 ANÁLISIS DE ESTABILIDAD:")
        print(f"\n   🔄 ESTABILIDAD EN CROSS-VALIDATION (menor std es mejor):")
        sorted_by_stability = df_results.sort_values('cv_roc_auc_std', ascending=True)
        for i, (model, row) in enumerate(sorted_by_stability.head(3).iterrows(), 1):
            medal = ['🥇', '🥈', '🥉'][i-1]
            cv_coeff = row['cv_roc_auc_std'] / row['cv_roc_auc_mean']
            print(f"      {medal} {model}: std={row['cv_roc_auc_std']:.4f} (CV={cv_coeff:.1%})")
        
        # Recomendaciones específicas
        print("\n💡 RECOMENDACIONES POR CASO DE USO:")
        
        best_accuracy = df_results['roc_auc'].idxmax()
        fastest_training = df_results['training_time'].idxmin()
        fastest_prediction = df_results['prediction_speed'].idxmax()
        most_stable = df_results['cv_roc_auc_std'].idxmin()
        best_balance = sorted_by_composite.index[0]
        
        print(f"   🎯 Máxima Precisión: {best_accuracy} (ROC-AUC: {df_results.loc[best_accuracy, 'roc_auc']:.4f})")
        print(f"   🚀 Entrenamiento Rápido: {fastest_training} ({df_results.loc[fastest_training, 'training_time']:.2f}s)")
        print(f"   ⚡ Predicción Rápida: {fastest_prediction} ({df_results.loc[fastest_prediction, 'prediction_speed']:.0f} pred/s)")
        print(f"   🔄 Más Estable: {most_stable} (CV std: {df_results.loc[most_stable, 'cv_roc_auc_std']:.4f})")
        print(f"   ⚖️ Mejor Balance: {best_balance} (Score: {sorted_by_composite.loc[best_balance, 'composite_score']:.3f})")
        
        return {
            'performance_ranking': df_results,
            'composite_ranking': sorted_by_composite,
            'recommendations': {
                'best_accuracy': best_accuracy,
                'fastest_training': fastest_training,
                'fastest_prediction': fastest_prediction,
                'most_stable': most_stable,
                'best_balance': best_balance
            }
        }
        
    def create_cost_benefit_analysis(self):
        """Crea análisis comparativo de costo-beneficio para selección de modelo"""
        if not self.results:
            print("❌ No hay resultados para analizar")
            return None
            
        print("\n💰 ANÁLISIS COSTO-BENEFICIO DE MODELOS")
        print("=" * 60)
        
        df_results = self.create_comparison_table()
        
        # Crear tabla comparativa ampliada
        comparison_data = []
        
        for model_name, row in df_results.iterrows():
            # Métricas de rendimiento
            roc_auc = row['roc_auc']
            precision = row['precision']
            recall = row['recall']
            f1_score = row['f1_score']
            ks_statistic = row['ks_statistic']
            
            # Métricas de costo/tiempo
            training_time = row['training_time']
            prediction_speed = row['prediction_speed']
            cv_std = row['cv_roc_auc_std']
            
            # Calcular scores normalizados (0-1)
            # Performance Score (promedio de métricas principales)
            performance_score = (roc_auc + precision + recall + f1_score + ks_statistic) / 5
            
            # Efficiency Score (velocidad vs tiempo)
            # Normalizar tiempos para score (invertir para que menor tiempo = mejor score)
            max_training_time = df_results['training_time'].max()
            min_training_time = df_results['training_time'].min()
            time_score = 1 - (training_time - min_training_time) / (max_training_time - min_training_time) if max_training_time != min_training_time else 1
            
            # Normalizar velocidad de predicción
            max_speed = df_results['prediction_speed'].max()
            min_speed = df_results['prediction_speed'].min()
            speed_score = (prediction_speed - min_speed) / (max_speed - min_speed) if max_speed != min_speed else 1
            
            efficiency_score = (time_score + speed_score) / 2
            
            # Stability Score (menor std es mejor)
            max_std = df_results['cv_roc_auc_std'].max()
            min_std = df_results['cv_roc_auc_std'].min()
            stability_score = 1 - (cv_std - min_std) / (max_std - min_std) if max_std != min_std else 1
            
            # Score compuesto final: 50% performance, 30% efficiency, 20% stability
            composite_score = (0.5 * performance_score + 0.3 * efficiency_score + 0.2 * stability_score)
            
            # Clasificación de costo computacional
            if training_time < 1.0:
                cost_category = "💚 BAJO"
            elif training_time < 5.0:
                cost_category = "🟡 MEDIO"
            else:
                cost_category = "🔴 ALTO"
                
            # Recomendación basada en uso
            if composite_score > 0.8 and training_time < 3.0:
                recommendation = "⭐ EXCELENTE"
            elif composite_score > 0.7 and training_time < 5.0:
                recommendation = "✅ RECOMENDADO"
            elif composite_score > 0.6:
                recommendation = "⚠️ ACEPTABLE"
            else:
                recommendation = "❌ NO RECOMENDADO"
            
            comparison_data.append({
                'Modelo': model_name,
                'ROC-AUC': f"{roc_auc:.4f}",
                'Precision': f"{precision:.4f}",
                'Recall': f"{recall:.4f}",
                'F1-Score': f"{f1_score:.4f}",
                'KS-Stat': f"{ks_statistic:.4f}",
                'T.Entrenamiento(s)': f"{training_time:.2f}",
                'Pred/seg': f"{prediction_speed:.0f}",
                'Estabilidad': f"{cv_std:.4f}",
                'Score Perf.': f"{performance_score:.3f}",
                'Score Efic.': f"{efficiency_score:.3f}",
                'Score Total': f"{composite_score:.3f}",
                'Costo Comp.': cost_category,
                'Recomendación': recommendation
            })
        
        # Crear DataFrame y ordenar por score total
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison['Score_Numeric'] = df_comparison['Score Total'].astype(float)
        df_comparison = df_comparison.sort_values('Score_Numeric', ascending=False)
        df_comparison = df_comparison.drop('Score_Numeric', axis=1)
        
        # Mostrar tabla completa
        print("\n📊 TABLA COMPARATIVA COMPLETA:")
        print("-" * 140)
        
        # Configurar display para mostrar toda la tabla
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print(df_comparison.to_string(index=False))
        
        # Análisis por categorías de uso
        print(f"\n🎯 RECOMENDACIONES POR CASO DE USO:")
        print("-" * 50)
        
        best_overall = df_comparison.iloc[0]
        fastest_training = df_comparison.loc[df_comparison['T.Entrenamiento(s)'].astype(float).idxmin()]
        best_precision = df_comparison.loc[df_comparison['Precision'].astype(float).idxmax()]
        most_balanced = best_overall  # Ya está ordenado por score total
        
        print(f"🏆 MEJOR GENERAL:")
        print(f"   • {best_overall['Modelo']}: Score {best_overall['Score Total']} | {best_overall['Recomendación']}")
        print(f"   • ROC-AUC: {best_overall['ROC-AUC']} | Tiempo: {best_overall['T.Entrenamiento(s)']}s | {best_overall['Costo Comp.']}")
        
        print(f"\n⚡ MÁS RÁPIDO DE ENTRENAR:")
        print(f"   • {fastest_training['Modelo']}: {fastest_training['T.Entrenamiento(s)']}s | Score {fastest_training['Score Total']}")
        print(f"   • ROC-AUC: {fastest_training['ROC-AUC']} | {fastest_training['Costo Comp.']}")
        
        print(f"\n🎯 MAYOR PRECISIÓN:")
        print(f"   • {best_precision['Modelo']}: Precision {best_precision['Precision']} | Score {best_precision['Score Total']}")
        print(f"   • Tiempo: {best_precision['T.Entrenamiento(s)']}s | {best_precision['Costo Comp.']}")
        
        # Análisis de trade-offs
        print(f"\n⚖️ ANÁLISIS DE TRADE-OFFS:")
        print("-" * 40)
        
        for idx, row in df_comparison.iterrows():
            training_cost = float(row['T.Entrenamiento(s)'].replace('s', ''))
            roc_auc = float(row['ROC-AUC'])
            
            if training_cost < 2.0 and roc_auc > 0.85:
                trade_off = "🎯 IDEAL: Alta precisión, bajo costo"
            elif training_cost < 1.0:
                trade_off = "💚 ECONÓMICO: Muy rápido de entrenar"
            elif roc_auc > 0.90:
                trade_off = "🏆 PREMIUM: Máxima precisión"
            elif training_cost > 5.0:
                trade_off = "💸 COSTOSO: Tiempo elevado"
            else:
                trade_off = "⚖️ BALANCEADO: Relación media"
                
            print(f"   • {row['Modelo']}: {trade_off}")
        
        # Recomendación final
        print(f"\n💡 RECOMENDACIÓN FINAL:")
        print("=" * 40)
        
        recommended_model = best_overall['Modelo']
        print(f"Para scoring crediticio se recomienda: {recommended_model}")
        print(f"Razones:")
        print(f"• Score general más alto: {best_overall['Score Total']}")
        print(f"• ROC-AUC competitivo: {best_overall['ROC-AUC']}")
        print(f"• Costo computacional: {best_overall['Costo Comp.']}")
        print(f"• Tiempo de entrenamiento: {best_overall['T.Entrenamiento(s)']}s")
        
        # Resetear configuración de pandas
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        
        return df_comparison
        
    def print_summary(self):
        """Imprime resumen ejecutivo"""
        if not self.results:
            print("❌ No hay resultados para mostrar")
            return
            
        df_results = self.create_comparison_table()
        best_model = self.get_best_model('roc_auc')
        
        print("=" * 80)
        print("🏆 RESUMEN EJECUTIVO - COMPARACIÓN DE MODELOS")
        print("=" * 80)
        
        print(f"\n📊 Modelos evaluados: {len(self.results)}")
        print(f"🥇 Mejor modelo general: {best_model['name']}")
        print(f"   ROC-AUC: {best_model['metrics']['roc_auc']:.4f}")
        print(f"   KS Statistic: {best_model['metrics']['ks_statistic']:.4f}")
        print(f"   F1-Score: {best_model['metrics']['f1_score']:.4f}")
        
        print("\n📈 TOP 3 POR ROC-AUC:")
        top_3 = df_results.head(3)
        for i, (name, row) in enumerate(top_3.iterrows(), 1):
            print(f"   {i}. {name}: {row['roc_auc']:.4f}")
        
        print("\n📋 TABLA COMPLETA DE RESULTADOS:")
        # Mostrar solo métricas principales para claridad
        display_cols = ['roc_auc', 'precision', 'recall', 'f1_score', 'ks_statistic']
        print(df_results[display_cols].to_string())

    def plot_performance_analysis(self, save_path="artifacts/performance_analysis.png"):
        """Genera gráficos específicos de análisis de performance"""
        if not self.results:
            print("❌ No hay resultados para visualizar")
            return
            
        df_results = self.create_comparison_table()
        
        # Configurar figura más grande para más gráficos
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        
        # 1. Performance vs Tiempo de Entrenamiento
        ax1 = axes[0, 0]
        scatter = ax1.scatter(df_results['training_time'], df_results['roc_auc'], 
                             s=100, alpha=0.7, c=range(len(df_results)), cmap='viridis')
        for i, model in enumerate(df_results.index):
            ax1.annotate(model, (df_results.loc[model, 'training_time'], 
                               df_results.loc[model, 'roc_auc']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax1.set_xlabel('Tiempo de Entrenamiento (s)')
        ax1.set_ylabel('ROC-AUC')
        ax1.set_title('Performance vs Tiempo de Entrenamiento')
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance vs Velocidad de Predicción
        ax2 = axes[0, 1]
        ax2.scatter(df_results['prediction_speed'], df_results['roc_auc'], 
                   s=100, alpha=0.7, c=range(len(df_results)), cmap='plasma')
        for i, model in enumerate(df_results.index):
            ax2.annotate(model, (df_results.loc[model, 'prediction_speed'], 
                               df_results.loc[model, 'roc_auc']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax2.set_xlabel('Velocidad de Predicción (pred/s)')
        ax2.set_ylabel('ROC-AUC')
        ax2.set_title('Performance vs Velocidad de Predicción')
        ax2.grid(True, alpha=0.3)
        
        # 3. Comparación de Tiempos
        ax3 = axes[0, 2]
        models = df_results.index
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, df_results['training_time'], width, 
                        label='Entrenamiento', alpha=0.8)
        bars2 = ax3.bar(x + width/2, df_results['prediction_time'] * 1000, width, 
                        label='Predicción (×1000)', alpha=0.8)
        
        ax3.set_xlabel('Modelos')
        ax3.set_ylabel('Tiempo (segundos)')
        ax3.set_title('Comparación de Tiempos')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Estabilidad (CV)
        ax4 = axes[0, 3]
        bars = ax4.bar(models, df_results['cv_roc_auc_std'])
        
        # Colorear por nivel de estabilidad
        for i, (bar, std_val) in enumerate(zip(bars, df_results['cv_roc_auc_std'])):
            if std_val < 0.005:
                bar.set_color('green')
            elif std_val < 0.01:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        ax4.set_xlabel('Modelos')
        ax4.set_ylabel('Desviación Estándar ROC-AUC')
        ax4.set_title('Estabilidad en Cross-Validation')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 5. Heatmap de Métricas Principales
        ax5 = axes[1, 0]
        metrics_for_heatmap = ['roc_auc', 'precision', 'recall', 'f1_score', 'ks_statistic']
        heatmap_data = df_results[metrics_for_heatmap].values
        
        im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')
        ax5.set_xticks(range(len(metrics_for_heatmap)))
        ax5.set_xticklabels([m.replace('_', ' ').title() for m in metrics_for_heatmap], rotation=45)
        ax5.set_yticks(range(len(models)))
        ax5.set_yticklabels(models)
        ax5.set_title('Heatmap de Métricas')
        
        # Añadir valores en el heatmap
        for i in range(len(models)):
            for j in range(len(metrics_for_heatmap)):
                text = ax5.text(j, i, f'{heatmap_data[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # 6. Comparación Multi-Métrica (Top 3)
        ax6 = axes[1, 1]
        
        # Seleccionar top 3 modelos y métricas principales
        top_3_models = df_results.head(3)
        metrics_comparison = ['roc_auc', 'precision', 'recall', 'f1_score', 'ks_statistic']
        
        # Crear gráfico de barras agrupadas
        x_pos = np.arange(len(metrics_comparison))
        width = 0.25
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (model_name, model_data) in enumerate(top_3_models.iterrows()):
            values = [model_data[metric] for metric in metrics_comparison]
            offset = (i - 1) * width
            bars = ax6.bar(x_pos + offset, values, width, 
                          label=model_name, color=colors[i], alpha=0.8)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=7)
        
        ax6.set_xlabel('Métricas')
        ax6.set_ylabel('Valor')
        ax6.set_title('Comparación Multi-Métrica - Top 3')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels([m.replace('_', ' ').title() for m in metrics_comparison], rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Score Compuesto
        ax7 = axes[1, 2]
        
        # Calcular score compuesto
        df_norm = df_results.copy()
        for col in ['roc_auc', 'precision', 'recall', 'f1_score']:
            df_norm[f'{col}_norm'] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())
        
        composite_score = df_norm[['roc_auc_norm', 'precision_norm', 'recall_norm', 'f1_score_norm']].mean(axis=1)
        
        bars = ax7.bar(models, composite_score)
        max_idx = composite_score.argmax()
        bars[max_idx].set_color('gold')
        
        ax7.set_xlabel('Modelos')
        ax7.set_ylabel('Score Compuesto')
        ax7.set_title('Score Compuesto de Performance')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
        
        # 8. Trade-off Performance vs Velocidad
        ax8 = axes[1, 3]
        
        # Crear un score de velocidad (combinando entrenamiento y predicción)
        speed_score = 1 / (df_results['training_time'] + 1/df_results['prediction_speed']*1000)
        
        ax8.scatter(speed_score, df_results['roc_auc'], s=150, alpha=0.7)
        
        for i, model in enumerate(df_results.index):
            ax8.annotate(model, (speed_score[model], df_results.loc[model, 'roc_auc']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax8.set_xlabel('Score de Velocidad (mayor es mejor)')
        ax8.set_ylabel('ROC-AUC')
        ax8.set_title('Trade-off: Performance vs Velocidad')
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Gráficos de performance guardados en: {save_path}")

def main():
    """Función principal para ejecutar la comparación"""
    print("🚀 Iniciando comparación comprehensiva de modelos...")
    
    # Cargar datos
    print("\n📁 Cargando dataset...")
    df = pd.read_csv("data/dataset_final_modelo.csv")
    
    X = df.drop("pago", axis=1)
    y = df["pago"]
    
    print(f"   Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"   Distribución: {dict(y.value_counts())}")
    
    # Configurar comparador
    comparator = ModelComparator()
    comparator.setup_default_models()
    
    # Ejecutar comparación
    results = comparator.compare_models(X, y)
    
    # Mostrar resultados
    comparator.print_summary()
    
    # Análisis de performance
    print("\n📊 Ejecutando análisis de performance...")
    performance_analysis = comparator.analyze_performance_comparison()
    
    # Análisis costo-beneficio
    print("\n💰 Ejecutando análisis costo-beneficio...")
    cost_benefit_analysis = comparator.create_cost_benefit_analysis()
    
    # Generar visualizaciones
    print("\n📊 Generando gráficos de comparación...")
    comparator.plot_comparison()
    
    # Generar gráficos de performance
    print("\n📈 Generando gráficos de performance...")
    comparator.plot_performance_analysis()
    
    # Guardar reportes
    print("\n💾 Guardando reportes...")
    comparator.save_comparison_report()
    
    # Guardar mejor modelo
    best_model = comparator.get_best_model('roc_auc')
    if best_model:
        model_path = f"artifacts/best_model_{best_model['name'].lower().replace(' ', '_')}.pkl"
        joblib.dump(best_model['model'], model_path)
        print(f"✅ Mejor modelo guardado en: {model_path}")
    
    print("\n🎉 Comparación completada!")
    print("📁 Archivos generados:")
    print("•artifacts/model_comparison_report.json")
    print("•artifacts/model_comparison.png")
    print("•artifacts/performance_analysis.png")
    print(f"•artifacts/best_model_{best_model['name'].lower().replace(' ', '_')}.pkl")

if __name__ == "__main__":
    main()