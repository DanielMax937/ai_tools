import pandas as pd
import glob
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution(series, stats_dict):
    """Create and save distribution plots"""
    non_null = series.dropna()
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram with KDE
    sns.histplot(data=non_null, bins=30, kde=True, ax=ax1)
    ax1.axvline(stats_dict['mean'], color='red', linestyle='--', label=f'Mean: {stats_dict["mean"]:.2f}')
    ax1.axvline(stats_dict['median'], color='green', linestyle='--', label=f'Median: {stats_dict["median"]:.2f}')
    ax1.set_title('Distribution of AI Results (Histogram with KDE)')
    ax1.set_xlabel('AI Result Score')
    ax1.set_ylabel('Count')
    ax1.legend()
    
    # Box plot
    sns.boxplot(data=non_null, ax=ax2)
    ax2.set_title('Box Plot of AI Results')
    ax2.set_xlabel('AI Result Score')
    
    # Add percentile annotations
    for percentile, value in [('P80', stats_dict['p80']), 
                            ('P90', stats_dict['p90']), 
                            ('P98', stats_dict['p98'])]:
        ax1.axvline(value, color='purple', linestyle=':', alpha=0.5)
        ax1.text(value, ax1.get_ylim()[1], f' {percentile}\n ({value:.1f})', 
                rotation=0, verticalalignment='top')
    
    # Add statistical information
    stats_text = (
        f"Statistical Summary:\n"
        f"Total Values: {len(non_null)}\n"
        f"Mean: {stats_dict['mean']:.2f}\n"
        f"Median: {stats_dict['median']:.2f}\n"
        f"Std Dev: {stats_dict['std']:.2f}\n"
        f"P80: {stats_dict['p80']:.2f}\n"
        f"P90: {stats_dict['p90']:.2f}\n"
        f"P98: {stats_dict['p98']:.2f}\n"
        f"Normal Distribution: {'Yes' if stats_dict['is_normal'] else 'No'}"
    )
    
    # Add text box with statistics
    plt.figtext(1.02, 0.5, stats_text, bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='center')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('ai_results_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

def analyze_results(series):
    """Calculate statistical measures for non-null values"""
    non_null = series.dropna()
    
    # Basic statistics
    mean = non_null.mean()
    median = non_null.median()
    std = non_null.std()
    
    # Percentiles
    p80 = np.percentile(non_null, 80)
    p90 = np.percentile(non_null, 90)
    p98 = np.percentile(non_null, 98)
    
    # Normal distribution test
    statistic, p_value = stats.normaltest(non_null)
    
    print("\nStatistical Analysis of AI Results:")
    print(f"Total non-null values: {len(non_null)}")
    print(f"\nNormal Distribution Test:")
    print(f"p-value: {p_value:.4f} (if > 0.05, data is normally distributed)")
    print(f"\nDistribution Parameters:")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Standard Deviation: {std:.2f}")
    print(f"\nPercentiles:")
    print(f"P80: {p80:.2f}")
    print(f"P90: {p90:.2f}")
    print(f"P98: {p98:.2f}")
    
    return {
        'mean': mean,
        'median': median,
        'std': std,
        'p80': p80,
        'p90': p90,
        'p98': p98,
        'is_normal': p_value > 0.05
    }

def merge_excel_results():
    # Define base file
    base_file = '20250630_classified_100rows_1.xlsx'
    
    # First read the base file
    try:
        base_df = pd.read_excel(base_file, sheet_name='Sheet1')
        print(f"\nBase file ({base_file}):")
        print(f"Columns: {base_df.columns.tolist()}")
        print(f"Number of rows: {len(base_df)}")
    except Exception as e:
        print(f"Error reading base file: {str(e)}")
        return
    
    # Get all xlsx files except the base file
    xlsx_files = [f for f in glob.glob('*.xlsx') if f != base_file and not f.startswith('merged')]
    
    # Initialize list to store all ai_results with their indices
    all_results = {}  # Using dict to preserve row indices
    
    print("\nProcessing other files:")
    # Process each file
    for file in xlsx_files:
        try:
            df = pd.read_excel(file, sheet_name='Sheet1')
            if 'ai_result' in df.columns:
                # Get non-null results with their indices
                non_null_results = df[df['ai_result'].notna()]
                
                if not non_null_results.empty:
                    print(f"\nFile: {file}")
                    print(f"Number of non-null results: {len(non_null_results)}")
                    
                    # Add results to our collection, using row index as key
                    for idx, row in non_null_results.iterrows():
                        all_results[idx] = row['ai_result']
            else:
                print(f"No ai_result column in {file}")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    if all_results:
        print(f"\nTotal non-null results collected: {len(all_results)}")
        
        # Add new results to base_df
        if 'ai_result' not in base_df.columns:
            base_df['ai_result'] = None
        
        # Fill in the ai_results using the correct indices
        for idx, result in all_results.items():
            if idx < len(base_df):
                base_df.loc[idx, 'ai_result'] = result
        
        # Analyze the results
        stats = analyze_results(base_df['ai_result'])
        
        # Create distribution plots
        plot_distribution(base_df['ai_result'], stats)
        print("\nDistribution plot saved as 'ai_results_distribution.png'")
        
        # Save the updated base file with a new name
        output_file = 'merged_with_base.xlsx'
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Copy the Data sheet from base file
            pd.read_excel(base_file, sheet_name='Data').to_excel(writer, sheet_name='Data', index=False)
            # Write the updated Sheet1
            base_df.to_excel(writer, sheet_name='Sheet1', index=False)
            
            # Create a new sheet for statistics
            stats_df = pd.DataFrame([stats])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        print(f"\nMerged results saved to {output_file}")
        print(f"Total rows in final file: {len(base_df)}")
        print(f"Total non-null AI results: {base_df['ai_result'].count()}")
    else:
        print("\nNo non-null ai_results found in any of the files")

if __name__ == "__main__":
    merge_excel_results() 