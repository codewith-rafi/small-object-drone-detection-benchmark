#!/usr/bin/env python3
"""
Interaction metrics calculator for P2 vs SAHI vs Both study.
Calculates synergy index, redundancy score, and interaction gain.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def calculate_interaction_metrics(baseline_results, sahi_results, p2_results, both_results):
    """
    Calculate interaction metrics between techniques.
    
    Args:
        baseline_results: dict with 'mAP50', 'mAP50_95', 'fps', 'small_object_ap'
        sahi_results: dict with same metrics for SAHI-only
        p2_results: dict with same metrics for P2-only
        both_results: dict with same metrics for P2+SAHI
        
    Returns:
        dict with calculated interaction metrics
    """
    
    print("="*80)
    print(" INTERACTION METRICS CALCULATION")
    print("="*80)
    
    # Extract metrics
    metrics = ['mAP50', 'mAP50_95', 'small_object_ap']
    
    interaction_results = {}
    
    for metric in metrics:
        print(f"\n📊 Analyzing: {metric}")
        
        # Get values
        B = baseline_results.get(metric, 0)
        S = sahi_results.get(metric, 0)
        P = p2_results.get(metric, 0)
        BOTH = both_results.get(metric, 0)
        
        print(f"  Baseline (B): {B:.4f}")
        print(f"  SAHI (S): {S:.4f} (Δ={S-B:+.4f})")
        print(f"  P2 (P): {P:.4f} (Δ={P-B:+.4f})")
        print(f"  Both: {BOTH:.4f} (Δ={BOTH-B:+.4f})")
        
        # Calculate individual gains
        gain_S = S - B
        gain_P = P - B
        gain_BOTH = BOTH - B
        
        # Expected additive gain (if independent)
        expected_additive = gain_S + gain_P
        
        # Interaction gain (actual - expected)
        interaction_gain = gain_BOTH - expected_additive
        
        # Synergy index (positive = synergistic, negative = redundant)
        synergy_index = interaction_gain / max(0.001, expected_additive)
        
        # Redundancy score (0-1, higher = more redundant)
        unique_contributions = abs(gain_S) + abs(gain_P)
        total_contribution = abs(gain_S) + abs(gain_P) + abs(interaction_gain)
        redundancy_score = 1 - (unique_contributions / max(0.001, total_contribution))
        
        # Overlap coefficient (how much techniques overlap)
        overlap_coefficient = min(abs(gain_S), abs(gain_P)) / max(0.001, max(abs(gain_S), abs(gain_P)))
        
        # Store results
        interaction_results[metric] = {
            'baseline': B,
            'sahi': S,
            'p2': P,
            'both': BOTH,
            'gain_sahi': gain_S,
            'gain_p2': gain_P,
            'gain_both': gain_BOTH,
            'expected_additive': expected_additive,
            'interaction_gain': interaction_gain,
            'synergy_index': synergy_index,
            'redundancy_score': redundancy_score,
            'overlap_coefficient': overlap_coefficient,
            'interpretation': interpret_interaction(synergy_index, redundancy_score)
        }
        
        print(f"  Expected additive: {expected_additive:.4f}")
        print(f"  Interaction gain: {interaction_gain:+.4f}")
        print(f"  Synergy index: {synergy_index:+.3f}")
        print(f"  Redundancy score: {redundancy_score:.3f}")
        print(f"  Overlap coefficient: {overlap_coefficient:.3f}")
        print(f"  Interpretation: {interpret_interaction(synergy_index, redundancy_score)}")
    
    # Calculate efficiency metrics (FPS trade-offs)
    print("\n⚡ Efficiency Analysis (FPS)")
    
    fps_B = baseline_results.get('fps', 1)
    fps_S = sahi_results.get('fps', 1)
    fps_P = p2_results.get('fps', 1)
    fps_BOTH = both_results.get('fps', 1)
    
    # FPS penalties
    fps_penalty_S = (fps_B - fps_S) / fps_B
    fps_penalty_P = (fps_B - fps_P) / fps_B
    fps_penalty_BOTH = (fps_B - fps_BOTH) / fps_B
    
    # Efficiency scores (mAP gain per FPS penalty)
    efficiency_S = (S - B) / max(0.001, fps_penalty_S)
    efficiency_P = (P - B) / max(0.001, fps_penalty_P)
    efficiency_BOTH = (BOTH - B) / max(0.001, fps_penalty_BOTH)
    
    interaction_results['efficiency'] = {
        'fps_baseline': fps_B,
        'fps_sahi': fps_S,
        'fps_p2': fps_P,
        'fps_both': fps_BOTH,
        'fps_penalty_sahi': fps_penalty_S,
        'fps_penalty_p2': fps_penalty_P,
        'fps_penalty_both': fps_penalty_BOTH,
        'efficiency_sahi': efficiency_S,
        'efficiency_p2': efficiency_P,
        'efficiency_both': efficiency_BOTH
    }
    
    print(f"  FPS: Baseline={fps_B:.1f}, SAHI={fps_S:.1f}, P2={fps_P:.1f}, Both={fps_BOTH:.1f}")
    print(f"  FPS penalties: SAHI={fps_penalty_S:.3f}, P2={fps_penalty_P:.3f}, Both={fps_penalty_BOTH:.3f}")
    print(f"  Efficiency (mAP gain/penalty): SAHI={efficiency_S:.3f}, P2={efficiency_P:.3f}, Both={efficiency_BOTH:.3f}")
    
    return interaction_results

def interpret_interaction(synergy_index, redundancy_score):
    """Interpret interaction based on metrics."""
    
    if synergy_index > 0.1:
        if redundancy_score < 0.3:
            return "Strong synergy - techniques complement each other well"
        else:
            return "Moderate synergy with some redundancy"
    
    elif synergy_index > -0.1:
        if redundancy_score > 0.7:
            return "Highly redundant - techniques recover similar information"
        elif redundancy_score > 0.4:
            return "Partially redundant - some overlap in benefits"
        else:
            return "Additive - independent benefits"
    
    else:  # synergy_index < -0.1
        if redundancy_score > 0.6:
            return "Strong interference - techniques conflict"
        else:
            return "Negative interaction - diminishing returns"

def generate_interaction_visualization(interaction_results, output_dir):
    """Generate visualization plots for interaction analysis."""
    
    print("\n" + "-"*80)
    print(" GENERATING INTERACTION VISUALIZATIONS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Synergy-Redundancy Quadrant Plot
    plt.figure(figsize=(10, 8))
    
    metrics = ['mAP50', 'mAP50_95', 'small_object_ap']
    colors = {'mAP50': 'blue', 'mAP50_95': 'green', 'small_object_ap': 'red'}
    
    for metric in metrics:
        if metric in interaction_results:
            data = interaction_results[metric]
            plt.scatter(data['redundancy_score'], data['synergy_index'],
                       c=colors[metric], s=200, alpha=0.7, label=metric)
            
            # Add text label
            plt.annotate(metric, 
                        (data['redundancy_score'], data['synergy_index']),
                        fontsize=10, ha='center', va='center')
    
    # Add quadrant lines
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
    
    # Add quadrant labels
    plt.text(0.25, 0.8, 'Synergistic\nLow Redundancy', 
             ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.5))
    plt.text(0.75, 0.8, 'Synergistic\nHigh Redundancy', 
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))
    plt.text(0.25, -0.8, 'Additive\nLow Redundancy', 
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.5))
    plt.text(0.75, -0.8, 'Interfering\nHigh Redundancy', 
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.5))
    
    plt.xlabel('Redundancy Score (Higher = More Redundant)')
    plt.ylabel('Synergy Index (Positive = Synergistic)')
    plt.title('Interaction Analysis: Synergy vs Redundancy')
    plt.xlim(0, 1)
    plt.ylim(-1, 1)
    plt.grid(True, alpha=0.3)
    
    quadrant_path = os.path.join(output_dir, 'interaction_quadrant.png')
    plt.savefig(quadrant_path, dpi=300, bbox_inches='tight')
    print(f"✅ Interaction quadrant plot saved to: {quadrant_path}")
    
    # Plot 2: Performance Gains Bar Chart
    plt.figure(figsize=(12, 6))
    
    metrics_list = []
    gain_sahi = []
    gain_p2 = []
    gain_both = []
    expected_additive = []
    
    for metric in metrics:
        if metric in interaction_results:
            data = interaction_results[metric]
            metrics_list.append(metric)
            gain_sahi.append(data['gain_sahi'])
            gain_p2.append(data['gain_p2'])
            gain_both.append(data['gain_both'])
            expected_additive.append(data['expected_additive'])
    
    x = np.arange(len(metrics_list))
    width = 0.2
    
    plt.bar(x - 1.5*width, gain_sahi, width, label='SAHI Gain', color='skyblue')
    plt.bar(x - 0.5*width, gain_p2, width, label='P2 Gain', color='lightgreen')
    plt.bar(x + 0.5*width, expected_additive, width, label='Expected Additive', 
            color='orange', alpha=0.7)
    plt.bar(x + 1.5*width, gain_both, width, label='Actual Both', color='coral')
    
    plt.xlabel('Metric')
    plt.ylabel('Gain over Baseline')
    plt.title('Performance Gains: Actual vs Expected')
    plt.xticks(x, metrics_list)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    gains_path = os.path.join(output_dir, 'performance_gains.png')
    plt.savefig(gains_path, dpi=300, bbox_inches='tight')
    print(f"✅ Performance gains plot saved to: {gains_path}")
    
    # Plot 3: Efficiency Trade-off
    plt.figure(figsize=(10, 6))
    
    if 'efficiency' in interaction_results:
        eff = interaction_results['efficiency']
        
        techniques = ['SAHI', 'P2', 'Both']
        mAP_gains = [interaction_results['mAP50']['gain_sahi'],
                     interaction_results['mAP50']['gain_p2'],
                     interaction_results['mAP50']['gain_both']]
        fps_penalties = [eff['fps_penalty_sahi'],
                         eff['fps_penalty_p2'],
                         eff['fps_penalty_both']]
        efficiencies = [eff['efficiency_sahi'],
                        eff['efficiency_p2'],
                        eff['efficiency_both']]
        
        plt.scatter(fps_penalties, mAP_gains, s=200, c=efficiencies, 
                   cmap='RdYlGn', alpha=0.7)
        
        # Add labels
        for i, tech in enumerate(techniques):
            plt.annotate(f"{tech}\nEff={efficiencies[i]:.2f}", 
                        (fps_penalties[i], mAP_gains[i]),
                        fontsize=10, ha='center', va='center')
        
        plt.colorbar(label='Efficiency (mAP gain / FPS penalty)')
        plt.xlabel('FPS Penalty (Higher = Slower)')
        plt.ylabel('mAP@0.5 Gain (Higher = Better)')
        plt.title('Efficiency Trade-off: Performance vs Speed')
        plt.grid(True, alpha=0.3)
        
        efficiency_path = os.path.join(output_dir, 'efficiency_tradeoff.png')
        plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
        print(f"✅ Efficiency trade-off plot saved to: {efficiency_path}")
    
    plt.close('all')

def save_interaction_report(interaction_results, output_dir):
    """Save comprehensive interaction report."""
    
    report_path = os.path.join(output_dir, 'interaction_analysis_report.json')
    
    with open(report_path, 'w') as f:
        json.dump(interaction_results, f, indent=2, default=str)
    
    print(f"✅ Interaction report saved to: {report_path}")
    
    # Also save summary CSV
    summary_data = []
    for metric, data in interaction_results.items():
        if metric != 'efficiency':
            summary_data.append({
                'metric': metric,
                'synergy_index': data['synergy_index'],
                'redundancy_score': data['redundancy_score'],
                'interaction_gain': data['interaction_gain'],
                'interpretation': data['interpretation']
            })
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, 'interaction_summary.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Interaction summary saved to: {csv_path}")
    
    # Print key findings
    print("\n" + "="*80)
    print(" KEY FINDINGS")
    print("="*80)
    
    for metric, data in interaction_results.items():
        if metric != 'efficiency':
            print(f"\n{metric.upper()}:")
            print(f"  Synergy: {data['synergy_index']:+.3f}")
            print(f"  Redundancy: {data['redundancy_score']:.3f}")
            print(f"  Interpretation: {data['interpretation']}")

def main():
    """Example main function with sample data."""
    
    print("INTERACTION METRICS CALCULATOR")
    print("="*80)
    
    # Example data structure (replace with actual results)
    example_results = {
        'baseline': {
            'mAP50': 0.560,
            'mAP50_95': 0.320,
            'small_object_ap': 0.280,
            'fps': 45.0
        },
        'sahi': {
            'mAP50': 0.620,
            'mAP50_95': 0.370,
            'small_object_ap': 0.350,
            'fps': 22.0
        },
        'p2': {
            'mAP50': 0.610,
            'mAP50_95': 0.360,
            'small_object_ap': 0.340,
            'fps': 38.0
        },
        'both': {
            'mAP50': 0.650,
            'mAP50_95': 0.390,
            'small_object_ap': 0.380,
            'fps': 18.0
        }
    }
    
    print("Using example data. Replace with actual experiment results.")
    print("\nTo use with actual data:")
    print("1. Run all four experiments (Baseline, SAHI, P2, Both)")
    print("2. Collect metrics for each condition")
    print("3. Update the example_results dictionary with your data")
    print("4. Run this script again")
    
    # Calculate interaction metrics
    interaction_results = calculate_interaction_metrics(
        example_results['baseline'],
        example_results['sahi'],
        example_results['p2'],
        example_results['both']
    )
    
    # Generate visualizations
    output_dir = 'interaction_analysis_results'
    generate_interaction_visualization(interaction_results, output_dir)
    
    # Save report
    save_interaction_report(interaction_results, output_dir)
    
    print("\n" + "="*80)
    print(" INTERACTION ANALYSIS COMPLETE")
    print("="*80)
    print("Next steps for paper:")
    print("1. Include quadrant plot in results section")
    print("2. Discuss synergy/redundancy findings in discussion")
    print("3. Reference efficiency trade-offs in practical guidelines")

if __name__ == '__main__':
    main()