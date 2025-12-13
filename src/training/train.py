from xgboost import XGBClassifier
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss, log_loss, cohen_kappa_score
)
from sklearn.calibration import calibration_curve
import json
import warnings
from lightgbm import LGBMClassifier
from overfitting_detector import run_complete_overfitting_analysis
warnings.filterwarnings('ignore')

def calculate_business_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calcula métricas de negocio específicas para scoring crediticio
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Métricas básicas
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Tasas específicas del negocio crediticio
    approval_rate = (tp + fp) / len(y_true)  # Tasa de aprobación
    default_rate = 1 - np.mean(y_true)  # Tasa de default general
    precision_good = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precisión en buenos pagadores
    recall_good = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensibilidad para buenos pagadores
    
    # KS Statistic (importante en scoring crediticio)
    bad_rate = []
    good_rate = []
    thresholds = np.linspace(0, 1, 100)
    
    for thresh in thresholds:
        pred_thresh = (y_pred_proba >= thresh).astype(int)
        if len(pred_thresh[pred_thresh == 1]) > 0:
            bad_captured = np.sum((y_true == 0) & (pred_thresh == 1)) / np.sum(y_true == 0)
            good_captured = np.sum((y_true == 1) & (pred_thresh == 1)) / np.sum(y_true == 1)
        else:
            bad_captured = 0
            good_captured = 0
        bad_rate.append(bad_captured)
        good_rate.append(good_captured)
    
    ks_stat = max(np.array(good_rate) - np.array(bad_rate))
    
    return {
        'approval_rate': approval_rate,
        'default_rate': default_rate,
        'precision_good_payers': precision_good,
        'recall_good_payers': recall_good,
        'ks_statistic': ks_stat,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }

def plot_model_evaluation(y_true, y_pred_proba, model=None, model_name="XGBoost"):
    """
    Genera gráficos para evaluación del modelo
    """
    plt.figure(figsize=(15, 12))
    
    # 1. ROC Curve
    plt.subplot(2, 3, 1)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    # 2. Precision-Recall Curve
    plt.subplot(2, 3, 2)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    # 3. Distribución de Scores
    plt.subplot(2, 3, 3)
    plt.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Buenos Pagadores', density=True)
    plt.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Malos Pagadores', density=True)
    plt.xlabel('Probabilidad de Pago')
    plt.ylabel('Densidad')
    plt.title('Distribución de Scores por Clase')
    plt.legend()
    plt.grid(True)
    
    # 4. Calibration Plot
    plt.subplot(2, 3, 4)
    fraction_positives, mean_predicted = calibration_curve(y_true, y_pred_proba, n_bins=10)
    plt.plot(mean_predicted, fraction_positives, "s-", label=model_name)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)
    
    # 5. Feature Importance (si está disponible)
    plt.subplot(2, 3, 5)
    if model is not None:
        try:
            # Asumiendo que el modelo tiene feature_importances_
            feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
            plt.title('Top 10 Feature Importances')
            plt.grid(True)
        except:
            plt.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Importance')
    else:
        plt.text(0.5, 0.5, 'Model not provided\nfor feature importance', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance')
    
    # 6. Confusion Matrix
    plt.subplot(2, 3, 6)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Añadir números a la matriz
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('artifacts/model_evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def comprehensive_model_evaluation(model, X_test, y_test, model_name="XGBoost"):
    """
    Evaluación comprehensiva del modelo con todas las métricas
    """
    # Predicciones
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Métricas estándar de clasificación
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'average_precision': average_precision_score(y_test, y_pred_proba),
        'log_loss': log_loss(y_test, y_pred_proba),
        'brier_score': brier_score_loss(y_test, y_pred_proba),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred)
    }
    
    # Métricas de negocio crediticio
    business_metrics = calculate_business_metrics(y_test, y_pred_proba)
    metrics.update(business_metrics)

    print(metrics)
    
    # Métricas por diferentes thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_metrics = {}
    
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        threshold_metrics[f'precision_at_{thresh}'] = precision_score(y_test, y_pred_thresh) if np.sum(y_pred_thresh) > 0 else 0
        threshold_metrics[f'recall_at_{thresh}'] = recall_score(y_test, y_pred_thresh)
        threshold_metrics[f'f1_at_{thresh}'] = f1_score(y_test, y_pred_thresh) if np.sum(y_pred_thresh) > 0 else 0
    
    metrics.update(threshold_metrics)
    
    return metrics, y_pred_proba

def cross_validation_evaluation(model, X, y, cv_folds=5):
    """
    Evaluación con validación cruzada
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Métricas a evaluar en CV
    cv_metrics = {
        'cv_accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
        'cv_precision': cross_val_score(model, X, y, cv=cv, scoring='precision'),
        'cv_recall': cross_val_score(model, X, y, cv=cv, scoring='recall'),
        'cv_f1': cross_val_score(model, X, y, cv=cv, scoring='f1'),
        'cv_roc_auc': cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    }
    
    # Estadísticas de CV
    cv_stats = {}
    for metric, scores in cv_metrics.items():
        cv_stats[f'{metric}_mean'] = np.mean(scores)
        cv_stats[f'{metric}_std'] = np.std(scores)
        cv_stats[f'{metric}_min'] = np.min(scores)
        cv_stats[f'{metric}_max'] = np.max(scores)
    
    return cv_stats

def save_model_report(metrics, cv_stats, model_name="XGBoost"):
    """
    Guarda un reporte completo del modelo en formato JSON
    """
    report = {
        'model_evaluation': metrics,
        'cross_validation': cv_stats,
        'timestamp': pd.Timestamp.now().isoformat(),
        'recommendations': generate_model_recommendations(metrics)
    }
    
    with open(f'artifacts/{model_name.lower()}_evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return report

def generate_model_recommendations(metrics):
    """
    Genera recomendaciones basadas en las métricas
    """
    recommendations = []
    
    if metrics['roc_auc'] < 0.7:
        recommendations.append("⚠️ ROC-AUC bajo (<0.7): Considera mejorar las features o probar otros algoritmos")
    elif metrics['roc_auc'] > 0.85:
        recommendations.append("✅ Excelente ROC-AUC (>0.85): El modelo discrimina muy bien")
    
    if metrics['ks_statistic'] < 0.2:
        recommendations.append("⚠️ KS Statistic bajo (<0.2): Poder de separación limitado")
    elif metrics['ks_statistic'] > 0.4:
        recommendations.append("✅ Excelente KS Statistic (>0.4): Muy buen poder de separación")
    
    if metrics['precision'] < 0.7:
        recommendations.append("⚠️ Precisión baja: Muchos falsos positivos (aprobar malos pagadores)")
    
    if metrics['recall'] < 0.7:
        recommendations.append("⚠️ Recall bajo: Perdiendo muchos buenos clientes")
    
    if metrics['brier_score'] > 0.25:
        recommendations.append("⚠️ Brier Score alto: Considera calibración del modelo")
    
    if len(recommendations) == 0:
        recommendations.append("✅ Modelo con buen rendimiento general")
    
    return recommendations

def print_model_summary(metrics, cv_stats):
    """
    Imprime un resumen ejecutivo del modelo
    """
    print("=" * 80)
    print(f"📊 EVALUACIÓN COMPLETA DEL MODELO: {metrics['model_name']}")
    print("=" * 80)
    
    print("\n🎯 MÉTRICAS PRINCIPALES:")
    print(f"  • ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  • Precisión: {metrics['precision']:.4f}")
    print(f"  • Recall: {metrics['recall']:.4f}")
    print(f"  • F1-Score: {metrics['f1_score']:.4f}")
    print(f"  • Accuracy: {metrics['accuracy']:.4f}")
    
    print("\n💼 MÉTRICAS DE NEGOCIO:")
    print(f"  • KS Statistic: {metrics['ks_statistic']:.4f}")
    print(f"  • Tasa de Aprobación: {metrics['approval_rate']:.2%}")
    print(f"  • Precisión Buenos Pagadores: {metrics['precision_good_payers']:.4f}")
    
    print("\n📈 VALIDACIÓN CRUZADA:")
    print(f"  • ROC-AUC (CV): {cv_stats['cv_roc_auc_mean']:.4f} ± {cv_stats['cv_roc_auc_std']:.4f}")
    print(f"  • F1-Score (CV): {cv_stats['cv_f1_mean']:.4f} ± {cv_stats['cv_f1_std']:.4f}")
    
    print("\n🔧 MÉTRICAS TÉCNICAS:")
    print(f"  • Log Loss: {metrics['log_loss']:.4f}")
    print(f"  • Brier Score: {metrics['brier_score']:.4f}")
    print(f"  • Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    
    print("\n📊 MATRIZ DE CONFUSIÓN:")
    print(f"  • Verdaderos Positivos: {metrics['true_positives']}")
    print(f"  • Falsos Positivos: {metrics['false_positives']}")
    print(f"  • Verdaderos Negativos: {metrics['true_negatives']}")
    print(f"  • Falsos Negativos: {metrics['false_negatives']}")

def main():
    print("🚀 Iniciando entrenamiento y evaluación comprehensiva del modelo...")
    
    # Cargar dataset
    print("\n📁 Cargando dataset...")
    df = pd.read_csv("../data/dataset_final_modelo.csv")
    print(f"   Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Información básica del dataset
    print(f"   Distribución de clases: {dict(df['pago'].value_counts())}")
    print(f"   Tasa de default: {(1 - df['pago'].mean()):.2%}")

    X = df.drop("pago", axis=1)
    y = df["pago"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"   Conjunto de prueba: {X_test.shape[0]} muestras")

    # Definir el modelo LightGBM
    print("\n🤖 Configurando modelo LightGBM...")
    model = LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    # Entrenar
    print("\n🔄 Entrenando modelo...")
    model.fit(X_train, y_train)
    print("   ✅ Entrenamiento completado")

    # Evaluación comprehensiva
    print("\n📊 Realizando evaluación comprehensiva...")
    metrics, y_pred_proba = comprehensive_model_evaluation(model, X_test, y_test, "LightGBM")
    
    # Validación cruzada
    print("\n🔀 Ejecutando validación cruzada...")
    cv_stats = cross_validation_evaluation(model, X, y, cv_folds=5)
    
    # Generar visualizaciones
    print("\n📈 Generando gráficos de evaluación...")
    plot_model_evaluation(y_test, y_pred_proba, model, "LightGBM")
    
    # Análisis de Overfitting
    print("\n🔍 Ejecutando análisis de overfitting...")
    overfitting_detector, overfitting_report = run_complete_overfitting_analysis(
        model, X, y, "LightGBM"
    )
    
    # Reporte completo
    print("\n💾 Guardando reporte de evaluación...")
    report = save_model_report(metrics, cv_stats, "LightGBM")
    
    # Añadir análisis de overfitting al reporte principal
    report['overfitting_analysis'] = overfitting_report
    
    # Imprimir resumen ejecutivo
    print_model_summary(metrics, cv_stats)
    
    # Recomendaciones
    print("\n💡 RECOMENDACIONES:")
    for rec in report['recommendations']:
        print(f"   {rec}")
    
    # Información de Overfitting
    print("\n🔍 ANÁLISIS DE OVERFITTING:")
    overfitting_risk = overfitting_report['overall_overfitting_risk']
    risk_colors = {'BAJO': '🟢', 'MEDIO': '🟡', 'ALTO': '🟠', 'CRÍTICO': '🔴'}
    print(f"   Riesgo de Overfitting: {risk_colors.get(overfitting_risk, '⚪')} {overfitting_risk}")
    
    # Mostrar top 3 señales de overfitting
    if overfitting_report['overfitting_signals']:
        print("   Top señales detectadas:")
        for signal in overfitting_report['overfitting_signals'][:3]:
            print(f"      {signal}")
    else:
        print("   ✅ No se detectaron señales preocupantes de overfitting")

    # Guardar modelo como artifact
    print("\n💾 Guardando modelo entrenado...")
    joblib.dump(model, "artifacts/g10tfm_model_test1.pkl")
    
    print("\nProceso completado exitosamente!")
    print("📁 Archivos generados:")
    print("• artifacts/g10tfm_model_test1.pkl (modelo entrenado)")
    print("• artifacts/lightgbm_evaluation_report.json (reporte completo)")
    print("• artifacts/model_evaluation_plots.png (gráficos de evaluación)")
    print("• artifacts/overfitting_analysis.png (análisis de overfitting)")
    print("• artifacts/overfitting_analysis_report.json (reporte de overfitting)")

if __name__ == "__main__":
    main()
