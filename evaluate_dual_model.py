"""
Dual-Model System: Performance Evaluation & Visualization
Generates confusion matrices, accuracy graphs, and performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                           recall_score, f1_score, roc_curve, auc, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# ============================================================================
# SAMPLE DATA (Replace with actual predictions from your models)
# ============================================================================

def generate_sample_data(n_samples=500, seed=42):
    """Generate sample predictions and ground truth"""
    np.random.seed(seed)
    
    # Lane 1: DR Detection (5 classes)
    dr_true = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    # Add some correlation to predictions
    dr_pred = dr_true.copy()
    # Introduce errors
    error_indices = np.random.choice(n_samples, int(n_samples * 0.15), replace=False)
    for idx in error_indices:
        dr_pred[idx] = np.random.choice([0, 1, 2, 3, 4])
    
    # Lane 2: Glaucoma/Cataract (3 classes)
    gc_true = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.25, 0.25])
    # Add some correlation to predictions
    gc_pred = gc_true.copy()
    # Introduce errors
    error_indices = np.random.choice(n_samples, int(n_samples * 0.1), replace=False)
    for idx in error_indices:
        gc_pred[idx] = np.random.choice([0, 1, 2])
    
    return {
        'lane1': {'y_true': dr_true, 'y_pred': dr_pred},
        'lane2': {'y_true': gc_true, 'y_pred': gc_pred}
    }


# ============================================================================
# PERFORMANCE METRICS CALCULATION
# ============================================================================

class DualModelEvaluator:
    """Evaluates dual-model system performance"""
    
    def __init__(self, data):
        self.data = data
        self.metrics = {}
        self.compute_metrics()
    
    def compute_metrics(self):
        """Compute all metrics for both lanes"""
        for lane_name, lane_data in self.data.items():
            y_true = lane_data['y_true']
            y_pred = lane_data['y_pred']
            
            self.metrics[lane_name] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_true, y_pred),
                'classification_report': classification_report(y_true, y_pred, zero_division=0),
            }
    
    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*80)
        print("DUAL-MODEL SYSTEM - PERFORMANCE EVALUATION")
        print("="*80)
        
        for lane_name, metrics in self.metrics.items():
            print(f"\n🚗 {lane_name.upper()}")
            print("-"*80)
            print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"\n  Classification Report:")
            print(f"  {metrics['classification_report']}")
    
    def get_metrics(self):
        """Return metrics dictionary"""
        return self.metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrices(evaluator, output_dir="./evaluation_results"):
    """Plot confusion matrices for both lanes"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = evaluator.get_metrics()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Lane 1: DR Detection
    cm_lane1 = metrics['lane1']['confusion_matrix']
    dr_classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    sns.heatmap(cm_lane1, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=dr_classes, yticklabels=dr_classes,
                cbar_kws={'label': 'Count'})
    axes[0].set_title('🚗 LANE 1: Diabetic Retinopathy Detection\nConfusion Matrix', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    
    # Lane 2: Glaucoma/Cataract
    cm_lane2 = metrics['lane2']['confusion_matrix']
    gc_classes = ['Normal', 'Glaucoma', 'Cataract']
    
    sns.heatmap(cm_lane2, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=gc_classes, yticklabels=gc_classes,
                cbar_kws={'label': 'Count'})
    axes[1].set_title('🚗 LANE 2: Glaucoma/Cataract Detection\nConfusion Matrix', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/confusion_matrices.png")
    plt.close()


def plot_accuracy_graphs(evaluator, output_dir="./evaluation_results"):
    """Plot accuracy and other metrics as bar graphs"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = evaluator.get_metrics()
    
    # Prepare data
    lanes = ['Lane 1\n(DR)', 'Lane 2\n(G/C)']
    accuracy = [metrics['lane1']['accuracy'], metrics['lane2']['accuracy']]
    precision = [metrics['lane1']['precision'], metrics['lane2']['precision']]
    recall = [metrics['lane1']['recall'], metrics['lane2']['recall']]
    f1 = [metrics['lane1']['f1'], metrics['lane2']['f1']]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#667eea', '#764ba2']
    
    # Accuracy
    axes[0, 0].bar(lanes, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('📊 Accuracy', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(accuracy):
        axes[0, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Precision
    axes[0, 1].bar(lanes, precision, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0, 1].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('🎯 Precision', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    for i, v in enumerate(precision):
        axes[0, 1].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[1, 0].bar(lanes, recall, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('🔍 Recall (Sensitivity)', fontsize=13, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(recall):
        axes[1, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # F1 Score
    axes[1, 1].bar(lanes, f1, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('⚖️ F1 Score', fontsize=13, fontweight='bold')
    axes[1, 1].set_ylim([0, 1])
    for i, v in enumerate(f1):
        axes[1, 1].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('🚗🚗 DUAL-MODEL PERFORMANCE METRICS', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/performance_metrics.png")
    plt.close()


def plot_per_class_accuracy(data, output_dir="./evaluation_results"):
    """Plot per-class accuracy for both lanes"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Lane 1: DR Detection
    lane1_data = data['lane1']
    y_true_l1 = lane1_data['y_true']
    y_pred_l1 = lane1_data['y_pred']
    
    dr_classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    per_class_acc_l1 = []
    
    for class_idx in range(5):
        mask = y_true_l1 == class_idx
        if mask.sum() > 0:
            acc = (y_pred_l1[mask] == class_idx).mean()
        else:
            acc = 0
        per_class_acc_l1.append(acc)
    
    colors_l1 = ['#10b981', '#f59e0b', '#ef6b42', '#dc2626', '#991b1b']
    axes[0].barh(dr_classes, per_class_acc_l1, color=colors_l1, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('🚗 LANE 1: Per-Class Accuracy (DR)\nDiabetic Retinopathy Detection', 
                      fontsize=13, fontweight='bold')
    axes[0].set_xlim([0, 1])
    for i, v in enumerate(per_class_acc_l1):
        axes[0].text(v + 0.02, i, f'{v:.2%}', va='center', fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Lane 2: Glaucoma/Cataract
    lane2_data = data['lane2']
    y_true_l2 = lane2_data['y_true']
    y_pred_l2 = lane2_data['y_pred']
    
    gc_classes = ['Normal', 'Glaucoma', 'Cataract']
    per_class_acc_l2 = []
    
    for class_idx in range(3):
        mask = y_true_l2 == class_idx
        if mask.sum() > 0:
            acc = (y_pred_l2[mask] == class_idx).mean()
        else:
            acc = 0
        per_class_acc_l2.append(acc)
    
    colors_l2 = ['#10b981', '#f59e0b', '#dc2626']
    axes[1].barh(gc_classes, per_class_acc_l2, color=colors_l2, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('🚗 LANE 2: Per-Class Accuracy (G/C)\nGlaucoma/Cataract Detection', 
                      fontsize=13, fontweight='bold')
    axes[1].set_xlim([0, 1])
    for i, v in enumerate(per_class_acc_l2):
        axes[1].text(v + 0.02, i, f'{v:.2%}', va='center', fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/per_class_accuracy.png")
    plt.close()


def plot_combined_risk_matrix(data, output_dir="./evaluation_results"):
    """Plot combined risk assessment matrix"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    lane1_data = data['lane1']
    lane2_data = data['lane2']
    
    # Create combined risk matrix (5x3)
    combined_matrix = np.zeros((5, 3))
    
    for i in range(len(lane1_data['y_true'])):
        dr_pred = lane1_data['y_pred'][i]
        gc_pred = lane2_data['y_pred'][i]
        combined_matrix[dr_pred, gc_pred] += 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    dr_classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    gc_classes = ['Normal', 'Glaucoma', 'Cataract']
    
    sns.heatmap(combined_matrix, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                xticklabels=gc_classes, yticklabels=dr_classes,
                cbar_kws={'label': 'Sample Count'})
    
    ax.set_xlabel('Lane 2 Prediction (Glaucoma/Cataract)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Lane 1 Prediction (Diabetic Retinopathy)', fontsize=12, fontweight='bold')
    ax.set_title('🚗🚗 COMBINED PREDICTION MATRIX\n(Lane 1 vs Lane 2 Predictions)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_risk_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/combined_risk_matrix.png")
    plt.close()


def plot_confidence_distribution(data, output_dir="./evaluation_results"):
    """Plot confidence score distributions"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Lane 1: Generate confidence scores
    lane1_data = data['lane1']
    n_samples_l1 = len(lane1_data['y_true'])
    correct_l1 = (lane1_data['y_true'] == lane1_data['y_pred'])
    confidence_l1_correct = np.random.uniform(0.7, 0.99, correct_l1.sum())
    confidence_l1_incorrect = np.random.uniform(0.3, 0.7, (~correct_l1).sum())
    
    axes[0].hist([confidence_l1_correct, confidence_l1_incorrect], 
                 bins=30, label=['Correct', 'Incorrect'],
                 color=['#10b981', '#dc2626'], alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('🚗 LANE 1: Confidence Distribution\n(DR Detection)', 
                      fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Lane 2: Generate confidence scores
    lane2_data = data['lane2']
    n_samples_l2 = len(lane2_data['y_true'])
    correct_l2 = (lane2_data['y_true'] == lane2_data['y_pred'])
    confidence_l2_correct = np.random.uniform(0.75, 0.99, correct_l2.sum())
    confidence_l2_incorrect = np.random.uniform(0.2, 0.65, (~correct_l2).sum())
    
    axes[1].hist([confidence_l2_correct, confidence_l2_incorrect], 
                 bins=30, label=['Correct', 'Incorrect'],
                 color=['#10b981', '#dc2626'], alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('🚗 LANE 2: Confidence Distribution\n(Glaucoma/Cataract Detection)', 
                      fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('🚗🚗 MODEL CONFIDENCE ANALYSIS', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confidence_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/confidence_distribution.png")
    plt.close()


def create_summary_report(evaluator, data, output_dir="./evaluation_results"):
    """Create a comprehensive summary report"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = evaluator.get_metrics()
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   DUAL-MODEL SYSTEM - EVALUATION REPORT                     ║
║                    Generated: {np.datetime64('today')}                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 OVERALL SYSTEM PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚗 LANE 1: DIABETIC RETINOPATHY DETECTION (Specialist)
───────────────────────────────────────────────────────
  • Accuracy:  {metrics['lane1']['accuracy']:.4f} ({metrics['lane1']['accuracy']*100:.2f}%)
  • Precision: {metrics['lane1']['precision']:.4f}
  • Recall:    {metrics['lane1']['recall']:.4f}
  • F1 Score:  {metrics['lane1']['f1']:.4f}
  
  Sample Size: {len(data['lane1']['y_true'])}
  Classes: 5 (No DR, Mild, Moderate, Severe, Proliferative)

🚗 LANE 2: GLAUCOMA/CATARACT DETECTION (Generalist)
──────────────────────────────────────────────────────
  • Accuracy:  {metrics['lane2']['accuracy']:.4f} ({metrics['lane2']['accuracy']*100:.2f}%)
  • Precision: {metrics['lane2']['precision']:.4f}
  • Recall:    {metrics['lane2']['recall']:.4f}
  • F1 Score:  {metrics['lane2']['f1']:.4f}
  
  Sample Size: {len(data['lane2']['y_true'])}
  Classes: 3 (Normal, Glaucoma, Cataract)

📈 COMBINED SYSTEM PERFORMANCE
────────────────────────────────
  Average Accuracy:  {(metrics['lane1']['accuracy'] + metrics['lane2']['accuracy'])/2:.4f}
  Average Precision: {(metrics['lane1']['precision'] + metrics['lane2']['precision'])/2:.4f}
  Average Recall:    {(metrics['lane1']['recall'] + metrics['lane2']['recall'])/2:.4f}
  Average F1 Score:  {(metrics['lane1']['f1'] + metrics['lane2']['f1'])/2:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 GENERATED VISUALIZATIONS
────────────────────────────
  ✅ confusion_matrices.png       - Confusion matrices for both lanes
  ✅ performance_metrics.png      - Accuracy, Precision, Recall, F1 Score
  ✅ per_class_accuracy.png       - Per-class accuracy breakdown
  ✅ combined_risk_matrix.png     - Combined predictions matrix
  ✅ confidence_distribution.png  - Confidence score distributions
  ✅ evaluation_summary.txt       - This report

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 KEY INSIGHTS
───────────────
  • Both models show {('high' if (metrics['lane1']['accuracy'] + metrics['lane2']['accuracy'])/2 > 0.8 else 'moderate')} accuracy
  • Lane 1 (DR): {'Specialist performing well' if metrics['lane1']['accuracy'] > 0.8 else 'Needs improvement'}
  • Lane 2 (G/C): {'Generalist performing well' if metrics['lane2']['accuracy'] > 0.8 else 'Needs improvement'}
  • System ready for {'production' if (metrics['lane1']['accuracy'] + metrics['lane2']['accuracy'])/2 > 0.85 else 'further refinement'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generated by: Dual-Model Evaluation System v2.0
"""
    
    with open(f'{output_dir}/evaluation_summary.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"✅ Saved: {output_dir}/evaluation_summary.txt")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete evaluation"""
    print("\n" + "="*80)
    print("🚗🚗 DUAL-MODEL SYSTEM - PERFORMANCE EVALUATION")
    print("="*80 + "\n")
    
    # Generate sample data
    print("📊 Generating sample data...")
    data = generate_sample_data(n_samples=500)
    print("✅ Sample data generated (500 samples)\n")
    
    # Initialize evaluator
    print("🔍 Computing performance metrics...")
    evaluator = DualModelEvaluator(data)
    print("✅ Metrics computed\n")
    
    # Print summary
    evaluator.print_summary()
    
    # Generate visualizations
    print("\n📈 Generating visualizations...\n")
    
    plot_confusion_matrices(evaluator)
    plot_accuracy_graphs(evaluator)
    plot_per_class_accuracy(data)
    plot_combined_risk_matrix(data)
    plot_confidence_distribution(data)
    create_summary_report(evaluator, data)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print("\n📁 All results saved to: ./evaluation_results/")
    print("\n🎯 Generated files:")
    print("   • confusion_matrices.png")
    print("   • performance_metrics.png")
    print("   • per_class_accuracy.png")
    print("   • combined_risk_matrix.png")
    print("   • confidence_distribution.png")
    print("   • evaluation_summary.txt")


if __name__ == "__main__":
    main()
