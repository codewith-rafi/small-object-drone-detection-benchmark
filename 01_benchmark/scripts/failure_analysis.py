#!/usr/bin/env python3
"""
Failure analysis for SAHI and P2 techniques.
Identifies when each technique fails and provides practical guidelines.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict

def analyze_failure_cases(baseline_predictions, sahi_predictions, p2_predictions, 
                         both_predictions, image_metadata):
    """
    Analyze failure cases across techniques.
    
    Args:
        baseline_predictions: dict of {image_id: [detections]}
        sahi_predictions: dict of {image_id: [detections]}
        p2_predictions: dict of {image_id: [detections]}
        both_predictions: dict of {image_id: [detections]}
        image_metadata: dict of {image_id: {'object_count': int, 'avg_size': float, 'density': float}}
        
    Returns:
        dict with failure analysis results
    """
    
    print("="*80)
    print(" FAILURE CASE ANALYSIS")
    print("="*80)
    
    results = {
        'technique_performance': {},
        'failure_patterns': {},
        'optimal_conditions': {},
        'practical_guidelines': {}
    }
    
    # Analyze each technique
    techniques = {
        'baseline': baseline_predictions,
        'sahi': sahi_predictions,
        'p2': p2_predictions,
        'both': both_predictions
    }
    
    # Calculate performance metrics per image
    for tech_name, predictions in techniques.items():
        print(f"\n📊 Analyzing: {tech_name.upper()}")
        
        tech_results = {
            'image_metrics': {},
            'failure_cases': [],
            'success_cases': []
        }
        
        for image_id, detections in predictions.items():
            metadata = image_metadata.get(image_id, {})
            gt_count = metadata.get('object_count', 0)
            
            if gt_count == 0:
                continue
            
            # Calculate metrics
            detected_count = len(detections)
            recall = detected_count / gt_count if gt_count > 0 else 0
            
            # Object size analysis (if available)
            avg_size = metadata.get('avg_size', 0)
            density = metadata.get('density', 0)
            
            # Store metrics
            tech_results['image_metrics'][image_id] = {
                'recall': recall,
                'detected_count': detected_count,
                'gt_count': gt_count,
                'avg_size': avg_size,
                'density': density
            }
            
            # Identify failure cases (recall < 0.3)
            if recall < 0.3:
                tech_results['failure_cases'].append({
                    'image_id': image_id,
                    'recall': recall,
                    'avg_size': avg_size,
                    'density': density,
                    'reason': classify_failure_reason(recall, avg_size, density, tech_name)
                })
            
            # Identify success cases (recall > 0.7)
            if recall > 0.7:
                tech_results['success_cases'].append({
                    'image_id': image_id,
                    'recall': recall,
                    'avg_size': avg_size,
                    'density': density
                })
        
        results['technique_performance'][tech_name] = tech_results
    
    # Compare techniques to identify patterns
    print("\n" + "-"*80)
    print(" FAILURE PATTERN ANALYSIS")
    print("-"*80)
    
    failure_patterns = analyze_failure_patterns(results)
    results['failure_patterns'] = failure_patterns
    
    # Determine optimal conditions for each technique
    print("\n" + "-"*80)
    print(" OPTIMAL CONDITIONS")
    print("-"*80)
    
    optimal_conditions = determine_optimal_conditions(results)
    results['optimal_conditions'] = optimal_conditions
    
    # Generate practical guidelines
    print("\n" + "-"*80)
    print(" PRACTICAL GUIDELINES")
    print("-"*80)
    
    guidelines = generate_guidelines(results)
    results['practical_guidelines'] = guidelines
    
    return results

def classify_failure_reason(recall, avg_size, density, technique):
    """Classify reason for failure based on image characteristics."""
    
    reasons = []
    
    # Size-based failures
    if avg_size < 32:  # Small objects
        if technique == 'baseline':
            reasons.append("small_objects_baseline")
        elif technique == 'sahi' and recall < 0.4:
            reasons.append("small_objects_sahi_ineffective")
    
    if avg_size > 96:  # Large objects
        if technique == 'p2' and recall < 0.4:
            reasons.append("large_objects_p2_ineffective")
    
    # Density-based failures
    if density > 0.8:  # High density
        if technique == 'sahi':
            reasons.append("high_density_sahi_failure")
    
    # General failures
    if recall < 0.2:
        reasons.append("severe_failure")
    elif recall < 0.4:
        reasons.append("moderate_failure")
    
    return reasons if reasons else ["unknown_failure"]

def analyze_failure_patterns(results):
    """Analyze patterns in failure cases across techniques."""
    
    patterns = {
        'common_failures': defaultdict(int),
        'technique_specific_failures': defaultdict(lambda: defaultdict(int)),
        'failure_clusters': []
    }
    
    # Analyze each technique
    for tech_name, tech_data in results['technique_performance'].items():
        failure_cases = tech_data['failure_cases']
        
        print(f"\n🔍 {tech_name.upper()} failure analysis:")
        print(f"  Total failure cases: {len(failure_cases)}")
        
        # Count failure reasons
        for case in failure_cases:
            for reason in case['reason']:
                patterns['common_failures'][reason] += 1
                patterns['technique_specific_failures'][tech_name][reason] += 1
        
        # Print top failure reasons
        if failure_cases:
            reason_counts = defaultdict(int)
            for case in failure_cases:
                for reason in case['reason']:
                    reason_counts[reason] += 1
            
            top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top failure reasons:")
            for reason, count in top_reasons:
                print(f"    - {reason}: {count} cases")
    
    # Identify failure clusters
    print("\n🔍 Failure clusters across techniques:")
    
    # Cluster 1: Small object failures
    small_object_failures = patterns['common_failures'].get('small_objects_baseline', 0)
    small_object_failures += patterns['common_failures'].get('small_objects_sahi_ineffective', 0)
    
    if small_object_failures > 0:
        patterns['failure_clusters'].append({
            'name': 'small_object_cluster',
            'description': 'Failures on small objects (<32px)',
            'count': small_object_failures,
            'affected_techniques': ['baseline', 'sahi'],
            'recommendation': 'Consider higher resolution or specialized small object detectors'
        })
        print(f"  Small object cluster: {small_object_failures} cases")
    
    # Cluster 2: High density failures
    density_failures = patterns['common_failures'].get('high_density_sahi_failure', 0)
    
    if density_failures > 0:
        patterns['failure_clusters'].append({
            'name': 'high_density_cluster',
            'description': 'Failures in high-density scenes',
            'count': density_failures,
            'affected_techniques': ['sahi'],
            'recommendation': 'Reduce SAHI overlap or use standard inference for crowded scenes'
        })
        print(f"  High density cluster: {density_failures} cases")
    
    # Cluster 3: Large object failures
    large_object_failures = patterns['common_failures'].get('large_objects_p2_ineffective', 0)
    
    if large_object_failures > 0:
        patterns['failure_clusters'].append({
            'name': 'large_object_cluster',
            'description': 'Failures on large objects (>96px)',
            'count': large_object_failures,
            'affected_techniques': ['p2'],
            'recommendation': 'P2 provides limited benefit for large objects'
        })
        print(f"  Large object cluster: {large_object_failures} cases")
    
    return patterns

def determine_optimal_conditions(results):
    """Determine optimal conditions for each technique."""
    
    optimal = {}
    
    for tech_name, tech_data in results['technique_performance'].items():
        success_cases = tech_data['success_cases']
        
        if not success_cases:
            optimal[tech_name] = {'conditions': [], 'confidence': 0}
            continue
        
        # Analyze characteristics of success cases
        sizes = [case['avg_size'] for case in success_cases]
        densities = [case['density'] for case in success_cases]
        recalls = [case['recall'] for case in success_cases]
        
        # Calculate optimal ranges
        size_mean = np.mean(sizes) if sizes else 0
        size_std = np.std(sizes) if len(sizes) > 1 else 0
        density_mean = np.mean(densities) if densities else 0
        density_std = np.std(densities) if len(densities) > 1 else 0
        
        optimal[tech_name] = {
            'optimal_size_range': {
                'min': max(0, size_mean - size_std),
                'max': size_mean + size_std,
                'mean': size_mean
            },
            'optimal_density_range': {
                'min': max(0, density_mean - density_std),
                'max': min(1, density_mean + density_std),
                'mean': density_mean
            },
            'success_rate': len(success_cases) / len(tech_data['image_metrics']) if tech_data['image_metrics'] else 0,
            'avg_recall_success': np.mean(recalls) if recalls else 0,
            'sample_size': len(success_cases)
        }
        
        print(f"\n✅ {tech_name.upper()} optimal conditions:")
        print(f"  Object size: {optimal[tech_name]['optimal_size_range']['min']:.1f}-"
              f"{optimal[tech_name]['optimal_size_range']['max']:.1f}px")
        print(f"  Scene density: {optimal[tech_name]['optimal_density_range']['min']:.2f}-"
              f"{optimal[tech_name]['optimal_density_range']['max']:.2f}")
        print(f"  Success rate: {optimal[tech_name]['success_rate']:.1%}")
        print(f"  Avg recall (success cases): {optimal[tech_name]['avg_recall_success']:.3f}")
    
    return optimal

def generate_guidelines(results):
    """Generate practical guidelines based on analysis."""
    
    guidelines = {
        'when_to_use_sahi': [],
        'when_to_use_p2': [],
        'when_to_use_both': [],
        'when_to_use_baseline': [],
        'technique_selection_flowchart': {}
    }
    
    # Analyze failure patterns
    patterns = results['failure_patterns']
    optimal = results['optimal_conditions']
    
    # Guidelines for SAHI
    sahi_failures = patterns['technique_specific_failures'].get('sahi', {})
    
    if sahi_failures.get('high_density_sahi_failure', 0) > 0:
        guidelines['when_to_use_sahi'].append(
            "Avoid SAHI in very crowded scenes (object density > 0.8)"
        )
    
    if optimal.get('sahi', {}).get('optimal_size_range', {}).get('mean', 0) < 50:
        guidelines['when_to_use_sahi'].append(
            "SAHI works best for small to medium objects (20-80px)"
        )
    
    guidelines['when_to_use_sahi'].append(
        "Use SAHI when inference speed is less critical than accuracy"
    )
    
    # Guidelines for P2
    p2_failures = patterns['technique_specific_failures'].get('p2', {})
    
    if p2_failures.get('large_objects_p2_ineffective', 0) > 0:
        guidelines['when_to_use_p2'].append(
            "P2 provides limited benefit for large objects (>96px)"
        )
    
    guidelines['when_to_use_p2'].append(
        "P2 is effective for small object detection across all scene densities"
    )
    
    guidelines['when_to_use_p2'].append(
        "Use P2 when you need consistent small object performance"
    )
    
    # Guidelines for Both
    both_success = optimal.get('both', {}).get('success_rate', 0)
    
    if both_success > 0.7:
        guidelines['when_to_use_both'].append(
            "Use Both when maximum accuracy is required regardless of speed"
        )
    
    guidelines['when_to_use_both'].append(
        "Both is computationally expensive but provides best small object recall"
    )
    
    # Guidelines for Baseline
    guidelines['when_to_use_baseline'].append(
        "Use Baseline when real-time inference is critical"
    )
    
    guidelines['when_to_use_baseline'].append(
        "Baseline works well for medium to large objects in moderate density scenes"
    )
    
    # Create selection flowchart
    guidelines['technique_selection_flowchart'] = {
        'step1': {
            'question': 'Is real-time inference critical?',
            'yes': 'Use Baseline',
            'no': 'Go to step 2'
        },
        'step2': {
            'question': 'Are objects primarily small (<32px)?',
            'yes': 'Go to step 3',
            'no': 'Consider Baseline or P2'
        },
        'step3': {
            'question': 'Is scene crowded (high object density)?',
            'yes': 'Use P2 (avoid SAHI)',
            'no': 'Go to step 4'
        },
        'step4': {
            'question': 'Can you tolerate slower inference?',
            'yes': 'Use Both for maximum accuracy',
            'no': 'Use SAHI for balanced performance'
        }
    }
    
    # Print guidelines
    print("\n📋 PRACTICAL GUIDELINES SUMMARY:")
    
    print("\n🔹 WHEN TO USE SAHI:")
    for guideline in guidelines['when_to_use_sahi']:
        print(f"  • {guideline}")
    
    print("\n🔹 WHEN TO USE P2:")
    for guideline in guidelines['when_to_use_p2']:
        print(f"  • {guideline}")
    
    print("\n🔹 WHEN TO USE BOTH:")
    for guideline in guidelines['when_to_use_both']:
        print(f"  • {guideline}")
    
    print("\n🔹 WHEN TO USE BASELINE:")
    for guideline in guidelines['when_to_use_baseline']:
        print(f"  • {guideline}")
    
    print("\n🔹 TECHNIQUE SELECTION FLOWCHART:")
    for step_name, step_data in guidelines['technique_selection_flowchart'].items():
        print(f"  {step_name}: {step_data['question']}")
        print(f"    → Yes: {step_data['yes']}")
        print(f"    → No: {step_data['no']}")
    
    return guidelines

def generate_failure_visualizations(results, output_dir):
    """Generate visualization plots for failure analysis."""
    
    print("\n" + "-"*80)
    print(" GENERATING FAILURE ANALYSIS VISUALIZATIONS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Failure reason distribution
    plt.figure(figsize=(12, 6))
    
    patterns = results['failure_patterns']
    common_failures = patterns['common_failures']
    
    if common_failures:
        reasons = list(common_failures.keys())
        counts = list(common_failures.values())
        
        bars = plt.barh(reasons, counts, color='lightcoral')
        plt.xlabel('Number of Failure Cases')
        plt.title('Common Failure Reasons Across All Techniques')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            plt.text(count, bar.get_y() + bar.get_height()/2, 
                    f' {count}', va='center')
    
    failure_dist_path = os.path.join(output_dir, 'failure_reason_distribution.png')
    plt.savefig(failure_dist_path, dpi=300, bbox_inches='tight')
    print(f"✅ Failure distribution plot saved to: {failure_dist_path}")
    
    # Plot 2: Technique performance by object size
    plt.figure(figsize=(10, 6))
    
    optimal = results['optimal_conditions']
    
    techniques = ['baseline', 'sahi', 'p2', 'both']
    colors = {'baseline': 'blue', 'sahi': 'green', 'p2': 'orange', 'both': 'red'}
    
    for tech in techniques:
        if tech in optimal and optimal[tech].get('optimal_size_range'):
            range_data = optimal[tech]['optimal_size_range']
            mean_size = range_data.get('mean', 0)
            success_rate = optimal[tech].get('success_rate', 0)
            
            plt.scatter(mean_size, success_rate, 
                       c=colors[tech], s=200, alpha=0.7, label=tech.upper())
            
            # Add error bars for size range
            plt.errorbar(mean_size, success_rate,
                        xerr=[[mean_size - range_data['min']], 
                              [range_data['max'] - mean_size]],
                        fmt='none', ecolor=colors[tech], alpha=0.5)
    
    plt.xlabel('Optimal Object Size (px)')
    plt.ylabel('Success Rate')
    plt.title('Technique Performance by Object Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    size_performance_path = os.path.join(output_dir, 'size_performance.png')
    plt.savefig(size_performance_path, dpi=300, bbox_inches='tight')
    print(f"✅ Size performance plot saved to: {size_performance_path}")
    
    plt.close('all')

def save_failure_analysis_report(results, output_dir):
    """Save comprehensive failure analysis report."""
    
    report_path = os.path.join(output_dir, 'failure_analysis_report.json')
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Failure analysis report saved to: {report_path}")
    
    # Save guidelines as separate markdown file
    guidelines_path = os.path.join(output_dir, 'practical_guidelines.md')
    
    with open(guidelines_path, 'w') as f:
        f.write("# Practical Guidelines for Technique Selection\n\n")
        
        f.write("## When to Use SAHI\n")
        for guideline in results['practical_guidelines']['when_to_use_sahi']:
            f.write(f"- {guideline}\n")
        
        f.write("\n## When to Use P2\n")
        for guideline in results['practical_guidelines']['when_to_use_p2']:
            f.write(f"- {guideline}\n")
        
        f.write("\n## When to Use Both\n")
        for guideline in results['practical_guidelines']['when_to_use_both']:
            f.write(f"- {guideline}\n")
        
        f.write("\n## When to Use Baseline\n")
        for guideline in results['practical_guidelines']['when_to_use_baseline']:
            f.write(f"- {guideline}\n")
        
        f.write("\n## Technique Selection Flowchart\n")
        for step_name, step_data in results['practical_guidelines']['technique_selection_flowchart'].items():
            f.write(f"### {step_name}\n")
            f.write(f"- Question: {step_data['question']}\n")
            f.write(f"- Yes: {step_data['yes']}\n")
            f.write(f"- No: {step_data['no']}\n\n")
    
    print(f"✅ Practical guidelines saved to: {guidelines_path}")

def main():
    """Example main function with sample data."""
    
    print("FAILURE ANALYSIS FOR SAHI AND P2 TECHNIQUES")
    print("="*80)
    
    print("\n⚠️  This is an example with synthetic data.")
    print("Replace with actual experiment results after running all conditions.")
    
    # Example data structure
    example_predictions = {}
    example_metadata = {}
    
    # Generate synthetic data for 100 images
    np.random.seed(42)
    
    for i in range(100):
        image_id = f"image_{i:03d}"
        
        # Generate random characteristics
        object_count = np.random.randint(5, 50)
        avg_size = np.random.uniform(20, 100)
        density = np.random.uniform(0.1, 0.9)
        
        # Store metadata
        example_metadata[image_id] = {
            'object_count': object_count,
            'avg_size': avg_size,
            'density': density
        }
        
        # Generate predictions for each technique
        # Baseline: struggles with small objects
        baseline_recall = 0.6 - (0.3 if avg_size < 32 else 0)
        example_predictions[f"baseline_{image_id}"] = {
            'detections': int(object_count * baseline_recall)
        }
        
        # SAHI: good for small objects, struggles with high density
        sahi_recall = 0.7 - (0.2 if density > 0.7 else 0)
        example_predictions[f"sahi_{image_id}"] = {
            'detections': int(object_count * sahi_recall)
        }
        
        # P2: good for small objects, consistent
        p2_recall = 0.65
        example_predictions[f"p2_{image_id}"] = {
            'detections': int(object_count * p2_recall)
        }
        
        # Both: best overall but slower
        both_recall = 0.75 - (0.1 if density > 0.8 else 0)
        example_predictions[f"both_{image_id}"] = {
            'detections': int(object_count * both_recall)
        }
    
    print("\nTo use with actual data:")
    print("1. Run inference on validation set for all 4 conditions")
    print("2. Collect per-image detection counts and ground truth")
    print("3. Update the data structures with your results")
    print("4. Run this script again")
    
    # Run analysis with example data
    output_dir = 'failure_analysis_results'
    
    # Note: In real usage, you would process the actual predictions
    # This example shows the structure but needs real data
    
    print(f"\n✅ Script ready. Output will be saved to: {output_dir}/")
    print("\nNext steps after collecting real data:")
    print("1. Update the data structures with your experiment results")
    print("2. Run the analysis to generate failure patterns")
    print("3. Use the guidelines in your paper discussion section")

if __name__ == '__main__':
    main()