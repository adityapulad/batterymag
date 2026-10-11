"""
Quick Start Script for Hybrid ECM-LSTM Models
Run all three predictions: SOC, SOH, and RUL

Usage:
    python run_all_hybrid.py
    python run_all_hybrid.py --battery B0006
    python run_all_hybrid.py --battery B0005 --skip-soc
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import argparse
import time
import pandas as pd
from datetime import datetime

# Import hybrid models
from hybrid_soc import run_hybrid_soc_analysis
from hybrid_soh import run_hybrid_soh_analysis
from hybrid_rul import run_hybrid_rul_analysis


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("🔋 HYBRID ECM-LSTM BATTERY ANALYSIS SUITE")
    print("="*70)
    print("Combining Physics-Based ECM with Data-Driven LSTM")
    print("Predictions: SOC | SOH | RUL")
    print("="*70 + "\n")


def print_section(title):
    """Print section header"""
    print("\n" + "─"*70)
    print(f"  {title}")
    print("─"*70 + "\n")


def run_complete_analysis(battery_id='B0005', skip_soc=False, skip_soh=False, skip_rul=False):
    """
    Run complete hybrid analysis for all three predictions
    
    Args:
        battery_id: Battery identifier
        skip_soc: Skip SOC analysis
        skip_soh: Skip SOH analysis
        skip_rul: Skip RUL analysis
    """
    print_banner()
    
    start_time = time.time()
    results = {}
    
    # Create results directory
    os.makedirs('hybrid/results', exist_ok=True)
    
    # ========== SOC ANALYSIS ==========
    if not skip_soc:
        print_section("1️⃣  SOC (State of Charge) Prediction")
        try:
            soc_start = time.time()
            hybrid_soc, metrics_soc = run_hybrid_soc_analysis(battery_id)
            soc_time = time.time() - soc_start
            
            results['SOC'] = {
                'metrics': metrics_soc,
                'time': soc_time,
                'status': 'SUCCESS'
            }
            
            print(f"\n✅ SOC Analysis Complete ({soc_time:.1f}s)")
        except Exception as e:
            print(f"\n❌ SOC Analysis Failed: {str(e)}")
            results['SOC'] = {'status': 'FAILED', 'error': str(e)}
    else:
        print_section("1️⃣  SOC Analysis - SKIPPED")
        results['SOC'] = {'status': 'SKIPPED'}
    
    # ========== SOH ANALYSIS ==========
    if not skip_soh:
        print_section("2️⃣  SOH (State of Health) Prediction")
        try:
            soh_start = time.time()
            hybrid_soh, metrics_soh, comparison_soh = run_hybrid_soh_analysis(battery_id)
            soh_time = time.time() - soh_start
            
            results['SOH'] = {
                'metrics': metrics_soh,
                'comparison': comparison_soh,
                'time': soh_time,
                'status': 'SUCCESS'
            }
            
            print(f"\n✅ SOH Analysis Complete ({soh_time:.1f}s)")
        except Exception as e:
            print(f"\n❌ SOH Analysis Failed: {str(e)}")
            results['SOH'] = {'status': 'FAILED', 'error': str(e)}
    else:
        print_section("2️⃣  SOH Analysis - SKIPPED")
        results['SOH'] = {'status': 'SKIPPED'}
    
    # ========== RUL ANALYSIS ==========
    if not skip_rul:
        print_section("3️⃣  RUL (Remaining Useful Life) Prediction")
        try:
            rul_start = time.time()
            hybrid_rul, metrics_rul, comparison_rul = run_hybrid_rul_analysis(battery_id)
            rul_time = time.time() - rul_start
            
            results['RUL'] = {
                'metrics': metrics_rul,
                'comparison': comparison_rul,
                'time': rul_time,
                'status': 'SUCCESS'
            }
            
            print(f"\n✅ RUL Analysis Complete ({rul_time:.1f}s)")
        except Exception as e:
            print(f"\n❌ RUL Analysis Failed: {str(e)}")
            results['RUL'] = {'status': 'FAILED', 'error': str(e)}
    else:
        print_section("3️⃣  RUL Analysis - SKIPPED")
        results['RUL'] = {'status': 'SKIPPED'}
    
    # ========== SUMMARY ==========
    total_time = time.time() - start_time
    print_summary(results, battery_id, total_time)
    
    # Save summary report
    save_summary_report(results, battery_id, total_time)
    
    return results


def print_summary(results, battery_id, total_time):
    """Print comprehensive summary"""
    print("\n" + "="*70)
    print("📊 ANALYSIS SUMMARY")
    print("="*70)
    print(f"Battery: {battery_id}")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print("="*70)
    
    # Create summary table
    summary_data = []
    
    for task in ['SOC', 'SOH', 'RUL']:
        if task in results and results[task]['status'] == 'SUCCESS':
            metrics = results[task]['metrics']
            
            if task == 'SOC':
                summary_data.append({
                    'Task': task,
                    'Status': '✅',
                    'RMSE': f"{metrics['RMSE']:.4f}%",
                    'MAE': f"{metrics['MAE']:.4f}%",
                    'R²': f"{metrics['R2']:.6f}",
                    'Time': f"{results[task]['time']:.1f}s"
                })
            elif task == 'SOH':
                summary_data.append({
                    'Task': task,
                    'Status': '✅',
                    'RMSE': f"{metrics['RMSE']:.4f}%",
                    'MAE': f"{metrics['MAE']:.4f}%",
                    'R²': f"{metrics['R2']:.6f}",
                    'Time': f"{results[task]['time']:.1f}s"
                })
            elif task == 'RUL':
                summary_data.append({
                    'Task': task,
                    'Status': '✅',
                    'RMSE': f"{metrics['RMSE']:.2f} cycles",
                    'MAE': f"{metrics['MAE']:.2f} cycles",
                    'R²': f"{metrics['R2']:.6f}",
                    'Time': f"{results[task]['time']:.1f}s"
                })
        elif task in results and results[task]['status'] == 'SKIPPED':
            summary_data.append({
                'Task': task,
                'Status': '⏭️',
                'RMSE': 'N/A',
                'MAE': 'N/A',
                'R²': 'N/A',
                'Time': 'N/A'
            })
        else:
            summary_data.append({
                'Task': task,
                'Status': '❌',
                'RMSE': 'FAILED',
                'MAE': 'FAILED',
                'R²': 'FAILED',
                'Time': 'N/A'
            })
    
    df = pd.DataFrame(summary_data)
    print("\n" + df.to_string(index=False))
    print("="*70)
    
    # Print improvements
    print("\n📈 IMPROVEMENTS OVER BASELINE LSTM:")
    print("-"*70)
    
    for task in ['SOH', 'RUL']:
        if task in results and results[task]['status'] == 'SUCCESS' and 'comparison' in results[task]:
            comp = results[task]['comparison']['improvement']
            print(f"\n{task}:")
            print(f"  RMSE: {comp['RMSE']:+.2f}%")
            print(f"  MAE:  {comp['MAE']:+.2f}%")
            print(f"  R²:   {comp['R2']:+.2f}%")
    
    print("\n" + "="*70)
    
    # Print saved files
    print("\n💾 SAVED FILES:")
    print("-"*70)
    print(f"Models:")
    if 'SOC' in results and results['SOC']['status'] == 'SUCCESS':
        print(f"  ✓ hybrid/best_hybrid_soc_model.h5")
    if 'SOH' in results and results['SOH']['status'] == 'SUCCESS':
        print(f"  ✓ hybrid/best_hybrid_soh_model.h5")
    if 'RUL' in results and results['RUL']['status'] == 'SUCCESS':
        print(f"  ✓ hybrid/best_hybrid_rul_model.h5")
    
    print(f"\nResults:")
    if 'SOC' in results and results['SOC']['status'] == 'SUCCESS':
        print(f"  ✓ hybrid/results/{battery_id}_hybrid_soc_results.png")
    if 'SOH' in results and results['SOH']['status'] == 'SUCCESS':
        print(f"  ✓ hybrid/results/{battery_id}_hybrid_soh_results.png")
    if 'RUL' in results and results['RUL']['status'] == 'SUCCESS':
        print(f"  ✓ hybrid/results/{battery_id}_hybrid_rul_results.png")
    
    print(f"\nReport:")
    print(f"  ✓ hybrid/results/{battery_id}_summary_report.txt")
    
    print("="*70)
    print("\n🎉 All analyses complete!")
    print("="*70 + "\n")


def save_summary_report(results, battery_id, total_time):
    """Save summary report to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_path = f'hybrid/results/{battery_id}_summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HYBRID ECM-LSTM ANALYSIS REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Battery ID: {battery_id}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
        f.write("="*70 + "\n\n")
        
        for task in ['SOC', 'SOH', 'RUL']:
            f.write(f"\n{task} PREDICTION:\n")
            f.write("-"*70 + "\n")
            
            if task in results and results[task]['status'] == 'SUCCESS':
                metrics = results[task]['metrics']
                f.write(f"Status: SUCCESS\n")
                f.write(f"Time: {results[task]['time']:.1f}s\n\n")
                
                f.write("Metrics:\n")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        if 'R2' in key or 'R²' in key:
                            f.write(f"  {key}: {value:.6f}\n")
                        elif 'RMSE' in key or 'MAE' in key:
                            if task == 'RUL':
                                f.write(f"  {key}: {value:.2f} cycles\n")
                            else:
                                f.write(f"  {key}: {value:.4f}%\n")
                        else:
                            f.write(f"  {key}: {value:.2f}\n")
                
                if 'comparison' in results[task]:
                    f.write("\nImprovement over Baseline:\n")
                    comp = results[task]['comparison']['improvement']
                    f.write(f"  RMSE: {comp['RMSE']:+.2f}%\n")
                    f.write(f"  MAE:  {comp['MAE']:+.2f}%\n")
                    f.write(f"  R²:   {comp['R2']:+.2f}%\n")
            
            elif task in results and results[task]['status'] == 'SKIPPED':
                f.write(f"Status: SKIPPED\n")
            else:
                f.write(f"Status: FAILED\n")
                if 'error' in results[task]:
                    f.write(f"Error: {results[task]['error']}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"\n✅ Summary report saved to: {report_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run Hybrid ECM-LSTM Analysis for SOC, SOH, and RUL'
    )
    parser.add_argument(
        '--battery',
        type=str,
        default='B0005',
        help='Battery ID (default: B0005)'
    )
    parser.add_argument(
        '--skip-soc',
        action='store_true',
        help='Skip SOC analysis'
    )
    parser.add_argument(
        '--skip-soh',
        action='store_true',
        help='Skip SOH analysis'
    )
    parser.add_argument(
        '--skip-rul',
        action='store_true',
        help='Skip RUL analysis'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    results = run_complete_analysis(
        battery_id=args.battery,
        skip_soc=args.skip_soc,
        skip_soh=args.skip_soh,
        skip_rul=args.skip_rul
    )
    
    return results


if __name__ == "__main__":
    main()
