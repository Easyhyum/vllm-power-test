#!/usr/bin/env python3
"""
배치 크기별 전력 소비 및 샘플당 에너지 시각화

사용법:
    python plot_power_energy.py logs/gpu_profile_164454.csv
    python plot_power_energy.py logs/gpu_profile_*.csv  # 여러 파일 분석
    python plot_power_energy.py logs/gpu_profile_*.csv --x-axis graph_batch_size  # graph_batch_size 기준
"""

import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_and_aggregate_data(csv_files, x_axis='batch_size'):
    """CSV 파일들을 로드하고 배치별로 집계
    
    Args:
        csv_files: CSV 파일 경로 리스트
        x_axis: 'batch_size' 또는 'graph_batch_size'
    """
    all_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
    
    if not all_data:
        raise ValueError("No valid CSV files found")
    
    # 모든 데이터 병합
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # cudagraph_mode 가 "FULL"인 행만 필터링
    combined_df = combined_df[combined_df['cudagraph_mode'] == 'FULL']

    # 샘플 개수 먼저 계산: index==1인 행의 개수 * batch_size
    # 각 그룹별로 index==1인 행 수를 세고 batch_size를 곱함
    sample_counts = combined_df[combined_df['index'] == combined_df['length']].groupby(x_axis).agg({
        'batch_size': lambda x: len(x) * x.iloc[0]  # 행 개수 * batch_size
    }).rename(columns={'batch_size': 'total_samples'})
    
    # index/length 비율이 0.5 이하인 항목 제거 (샘플 개수 계산 후에 적용)
    combined_df = combined_df[combined_df['index'] / combined_df['length'] > 0.5]
    
    # 배치별로 집계 (x_axis 컬럼은 제외)
    agg_dict = {
        'power': ['mean', 'std', 'min', 'max'],
        'avg_power_js': 'first',
        'total_energy_j': 'first',
        'throughput': 'first',
        'gpu_util': 'mean',
        'temperature': 'mean',
        'during_time': 'mean'
    }
    
    # x_axis가 아닌 컬럼만 추가
    other_axis = 'graph_batch_size' if x_axis == 'batch_size' else 'batch_size'
    agg_dict[other_axis] = 'first'
    combined_df = combined_df[combined_df[x_axis].isin([1,2,4,5,6,7,8,9,10,16,24])]
    grouped = combined_df.groupby(x_axis).agg(agg_dict).reset_index()
    
    # 컬럼명 정리 (멀티인덱스 컬럼 평탄화)
    new_columns = [x_axis]
    for col in grouped.columns[1:]:
        if isinstance(col, tuple):
            if col[1]:  # 집계 함수가 있는 경우
                if col[0] == 'power':
                    new_columns.append(f'power_{col[1]}')
                else:
                    new_columns.append(col[0])
            else:
                new_columns.append(col[0])
        else:
            new_columns.append(col)
    
    grouped.columns = new_columns
    
    # x_axis가 아닌 컬럼이 없으면 x_axis 값으로 채우기
    if other_axis not in grouped.columns:
        grouped[other_axis] = grouped[x_axis]
    
    # 컬럼명 명시적 매핑
    column_mapping = {
        'avg_power_js': 'avg_power',
        'total_energy_j': 'total_energy'
    }
    grouped = grouped.rename(columns=column_mapping)
    
    # 샘플 개수 병합 (inner join으로 변경하여 데이터가 있는 그룹만 유지)
    grouped = grouped.merge(sample_counts, on=x_axis, how='inner')
    
    # 샘플당 에너지 재계산
    grouped['energy_per_sample'] = grouped['total_energy'] / grouped['total_samples']
    
    # NaN이 있는 행 제거
    grouped = grouped.dropna(subset=['total_samples', 'energy_per_sample'])
    
    return grouped.sort_values(x_axis)

def plot_power_and_energy(data, output_file='power_energy_plot.png', x_axis='batch_size'):
    """전력과 에너지를 이중 축으로 시각화
    
    Args:
        data: 집계된 데이터프레임
        output_file: 출력 파일명
        x_axis: X축으로 사용할 컬럼 ('batch_size' 또는 'graph_batch_size')
    """
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    batch_sizes = data[x_axis].values
    x_pos = np.arange(len(batch_sizes))
    
    # X축 라벨 결정
    x_label = 'Batch Size' if x_axis == 'batch_size' else 'Graph Batch Size'
    
    # 왼쪽 축: 평균 전력 (histogram/bar)
    color1 = '#2E86AB'
    bars = ax1.bar(x_pos, data['power_mean'], 
                   yerr=data['power_std'],
                   alpha=0.7, 
                   color=color1,
                   edgecolor='black',
                   linewidth=1.5,
                   capsize=5,
                   label='Average Power (W)')
    
    ax1.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Power (W)', fontsize=14, fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{int(bs)}' for bs in batch_sizes])
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, data['power_mean'].max() * 1.3)

    # 막대 위에 정확한 값 표시
    for i, (bar, val, std) in enumerate(zip(bars, data['power_mean'], data['power_std'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}W\n±{std:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 오른쪽 축: 샘플당 에너지 (line graph)
    ax2 = ax1.twinx()
    color2 = '#A23B72'
    j_p_sample = data['energy_per_sample'] / 600  # y축 스케일 조정
    line = ax2.plot(x_pos, j_p_sample, 
                    color=color2, 
                    marker='o', 
                    markersize=10,
                    linewidth=3,
                    label='Energy per Sample (J/sample)')
    
    ax2.set_ylabel('Energy per Sample (J/sample)', fontsize=14, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
    
    # 라인 위에 정확한 값 표시
    for i, (x, y) in enumerate(zip(x_pos, j_p_sample)):
        ax2.text(x, y, f'{y:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=color2)
    
    # 제목 및 범례
    title = f'Power Consumption and Energy Efficiency by {x_label}\n(LLaMA 3.1-8B-Instruct Inference)'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 범례 통합
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='lower left', fontsize=11, framealpha=0.9)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 파일 저장
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    # 화면 표시
    plt.show()

def print_summary_table(data, x_axis='batch_size'):
    """요약 테이블 출력
    
    Args:
        data: 집계된 데이터프레임
        x_axis: X축으로 사용할 컬럼
    """
    x_label = 'Batch' if x_axis == 'batch_size' else 'Graph'
    
    print("\n" + "="*120)
    print(f"{x_label} 크기별 성능 요약")
    print("="*120)
    print(f"{x_label:>6} | {'Batch':>6} | {'Graph':>6} | {'Samples':>8} | {'Avg Power':>10} | "
          f"{'Power StdDev':>12} | {'Energy/Sample':>15} | {'Throughput':>12} | {'GPU Util':>9} | {'Temp':>7}")
    print("-"*120)
    
    for _, row in data.iterrows():
        if x_axis == 'batch_size':
            print(f"{int(row[x_axis]):6d} | "
                f"{int(row['batch_size']):6d} | "
                f"{int(row['total_samples']):8d} | "
                f"{row['power_mean']:10.2f}W | "
                f"{row['power_std']:12.2f}W | "
                f"{row['energy_per_sample'] / 600:15.6f}J | "
                f"{row['throughput']:12.2f}/s | "
                f"{row['gpu_util']:9.1f}% | "
                f"{row['temperature']:7.1f}°C")
        else:
            print(f"{int(row[x_axis]):6d} | "
                f"{int(row['batch_size']):6d} | "
                f"{int(row['graph_batch_size']):6d} | "
                f"{row['power_mean']:10.2f}W | "
                f"{row['power_std']:12.2f}W | "
                f"{row['energy_per_sample'] / 600:15.6f}J | "
                f"{row['throughput']:12.2f}/s | "
                f"{row['gpu_util']:9.1f}% | "
                f"{row['temperature']:7.1f}°C")

    print("="*120)
    
    # 효율성 분석
    best_power_efficiency = data.loc[(data['energy_per_sample']/600).idxmin()]
    print(f"\n✓ 최고 에너지 효율: {x_label} {int(best_power_efficiency[x_axis])} "
          f"({best_power_efficiency['energy_per_sample'] / 600:.6f} J/sample, "
          f"{int(best_power_efficiency['total_samples'])} samples)")
    
    highest_throughput = data.loc[data['throughput'].idxmax()]
    print(f"✓ 최고 처리량: {x_label} {int(highest_throughput[x_axis])} "
          f"({highest_throughput['throughput']:.2f} samples/s)")
    print()

def main():
    parser = argparse.ArgumentParser(
        description='배치 크기별 전력 소비 및 샘플당 에너지 시각화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_power_energy.py logs/gpu_profile_164454.csv
  python plot_power_energy.py logs/gpu_profile_*.csv
  python plot_power_energy.py logs/gpu_profile_*.csv --x-axis graph_batch_size
  python plot_power_energy.py logs/gpu_profile_*.csv -x graph_batch_size -o output.png
        """
    )
    
    parser.add_argument('csv_pattern', help='CSV 파일 경로 또는 패턴 (예: logs/gpu_profile_*.csv)')
    parser.add_argument('-x', '--x-axis', 
                       choices=['batch_size', 'graph_batch_size'],
                       default='batch_size',
                       help='X축으로 사용할 값 (default: batch_size)')
    parser.add_argument('-o', '--output',
                       default='power_energy_analysis.png',
                       help='출력 파일명 (default: power_energy_analysis.png)')
    
    args = parser.parse_args()
    
    # CSV 파일 찾기
    csv_files = glob.glob(args.csv_pattern)
    
    if not csv_files:
        print(f"Error: No CSV files found matching pattern: {args.csv_pattern}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f}")
    
    print(f"\nX-axis: {args.x_axis}")
    
    # 데이터 로드 및 집계
    print("Loading and aggregating data...")
    data = load_and_aggregate_data(csv_files, x_axis=args.x_axis)
    
    # 요약 테이블 출력
    print_summary_table(data, x_axis=args.x_axis)
    
    # 플롯 생성
    print(f"\nGenerating plot...")
    plot_power_and_energy(data, args.output, x_axis=args.x_axis)
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
