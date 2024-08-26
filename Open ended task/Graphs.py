import Utils
import Results
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import itertools

color_palette = 'hsv'
# color_palette = 'RdPu'

bad_model_names = []
# bad_model_names = ['bart', 'inferSent', 'sbert']
model_names = Utils.model_names[:]
for bad_model_name in bad_model_names:
    model_names.remove(bad_model_name)

os.makedirs('Graphs', exist_ok=True)

# Compare data to corresponding normal distribution with a histogram & a QQ plot
def plot_histogram_and_qq(data):
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, ax = plt.subplots(1, 2, figsize=(4, 2))

    # Histogram with normal distribution curve
    sns.histplot(data, kde=False, stat='density', ax=ax[0], color='blue', bins=20)
    
    # Fit a normal distribution to the data
    mu, std = stats.norm.fit(data)
    xmin, xmax = ax[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    
    # Plot the normal distribution curve
    ax[0].plot(x, p, 'r', linewidth=3)
    ax[0].set_title('Histogram with Normal Distribution')
    ax[0].set_xlabel('Value')
    ax[0].set_ylabel('Density')

    # QQ plot
    stats.probplot(data, dist="norm", plot=ax[1])
    ax[1].set_title('QQ Plot')

    # Adjust layout
    plt.tight_layout()
    plt.show()
    plt.close()

# Show a regression in a subplot
def plot_regression_subplot(ax, points, datasets, dataset1_name, dataset2_name):
    # Extract the data
    names = list(points.keys())
    x_values = [points[name][0] for name in names]
    y_values = [points[name][1] for name in names]
    
    # Create the scatter plot
    palette = sns.color_palette(color_palette, len(names))  # Use unique colors
    
    # Plot each point with a unique color
    for i, name in enumerate(names):
        ax.scatter(x_values[i], y_values[i], color=palette[i], label=name, s=75)  # s controls the size of the points
    
    # Calculate regression statistics
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    X = sm.add_constant(x_values)
    model = sm.OLS(y_values, X).fit()
    slope, intercept = model.params[1], model.params[0]
    r_squared = model.rsquared
    
    # Choose regression line color based on R² value
    reg_color = 'red' if r_squared >= 0.5 else 'blue'
    
    # Add the regression line (without confidence interval)
    sns.regplot(x=x_values, y=y_values, scatter=False, color=reg_color, ci=None, ax=ax)
    
    # Add labels and title
    ax.set_xlabel(dataset1_name)
    ax.set_ylabel(dataset2_name)
    ax.set_title(f'{dataset1_name} ({datasets[dataset1_name]['type']}s) vs {dataset2_name} ({datasets[dataset2_name]['type']}s)', fontsize=11)
    
    # Set axis ticks to only 6, with nice rounded values
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    
    # Add regression statistics text without a box
    stats_text = f'Slope: {slope:.2f}, R²: {r_squared:.2f}'
    ax.text(0.975, 0.0025, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right')

# Show the regression of model performances for each pair of datasets
def plot_all_regressions_separated(datasets, results, figsize=(13, 7), save=True):
    # Create two lists to store points for high and low R² values
    high_r_squared_points = []
    low_r_squared_points = []
    
    # Create a dictionary to store colors for the legend
    colors = {}
    for i, name in enumerate(model_names):
        colors[name] = sns.color_palette(color_palette, len(model_names))[i]
    
    # Populate the lists based on R² values
    for dataset1_name, dataset2_name in itertools.combinations(Utils.dataset_names, 2):
        points = dict()
        for model_name in model_names:
            score1 = Results.get_result(results, datasets, dataset1_name, model_name)[0]
            score2 = Results.get_result(results, datasets, dataset2_name, model_name)[0]
            points[model_name] = (score1, score2)
        
        # Calculate regression statistics to determine R² value
        x_values = [points[name][0] for name in points]
        y_values = [points[name][1] for name in points]
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        X = sm.add_constant(x_values)
        model = sm.OLS(y_values, X).fit()
        r_squared = model.rsquared
        
        # Store the dataset names and points based on R² threshold
        if r_squared > 0.15:
            high_r_squared_points.append((dataset1_name, dataset2_name, points))
        else:
            low_r_squared_points.append((dataset1_name, dataset2_name, points))
    
    # Define a function to plot the regressions based on points
    def plot_regressions(points_list, title, suffix):
        num_plots = len(points_list)
        num_cols = 3  # Define the number of columns you want in the grid
        num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()  # Flatten the axes array to iterate over it easily
        
        # Plot each regression in a subplot
        for i, (dataset1_name, dataset2_name, points) in enumerate(points_list):
            plot_regression_subplot(axes[i], points, datasets, dataset1_name, dataset2_name)
        
        # Hide any unused subplots
        for j in range(i + 1, num_rows * num_cols):
            fig.delaxes(axes[j])
        
        # Add a legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=name) for name, color in colors.items()]
        fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.7), title='Embedding Models')
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        # Adjust the layout
        fig.tight_layout(rect=[0, 0, 0.98, 0.98])  # Adjust layout to make room for the legend and title
        # Save the figure if a filename prefix is provided
        if save:
            plt.savefig(f'Graphs/Regressions_{suffix}.png', bbox_inches='tight')
        plt.show()
        plt.close()

    # Plot regressions for high R²
    plot_regressions(high_r_squared_points, 'Non-Low R² Regressions (R² > 0.15)','High')
    
    # Plot regressions for low R²
    plot_regressions(low_r_squared_points, 'Low R² Regressions (R² ≤ 0.15)', 'Low')


# Show bar plot with comparisons to baseline
def plot_dataset_comparison(datasets, results, filename=None):
    # Set up the grid with 3 columns
    num_datasets = len(datasets)
    num_cols = 3
    num_rows = (num_datasets + num_cols - 1) // num_cols  # Calculate the number of rows needed
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 3 * num_rows))  # Adjust figure size accordingly
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    palette = sns.color_palette(color_palette, len(Utils.model_names))  # Use unique colors
    
    # Iterate over each dataset
    for i, dataset_name in enumerate(datasets):
        ax = axes[i]  # Get the current axis
        model_scores = []
        baseline_scores = []
        annotations = []
        
        # Gather the scores and annotations for each model
        for model_name in Utils.model_names:
            model_result, baseline_result, total_significant, num_hypotheses = Results.get_result(results, datasets, dataset_name, model_name)
            model_scores.append(model_result)
            baseline_scores.append(baseline_result)
            annotations.append(f'{total_significant}/{num_hypotheses}')
        
        # Locations for the bars
        ind = np.arange(len(Utils.model_names))
        width_total = 0.7
        width_model = 0.5
        width_baseline = width_total-width_model
        
        # Plot the model bars (move the labels a bit to the left)
        bars_model = ax.bar(ind, model_scores, width_model, color=palette)
        bars_model_labels = ax.bar(ind-width_model/3.5, model_scores, width_model, alpha=0)
        
        # Plot the baseline bars
        bars_baseline = ax.bar(ind + (width_model+width_baseline) / 2, baseline_scores, width_baseline, color='gray', alpha=0.75)
        
        # Add annotations above the model bars
        ax.bar_label(bars_model_labels, labels=annotations, padding=3, rotation=-90, color='black', fontsize=9)
        
        # Set the x-ticks and labels
        ax.set_xticks(ind + (width_model+width_baseline) / 4)
        ax.set_xticklabels(Utils.model_names, rotation=45, ha='right')
        
        # Set the title of the subplot to the dataset name
        ax.set_title(dataset_name)

        # Logscale
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        
    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Set the overall title for the grid
    fig.suptitle('Model-Baseline Comparisons per Dataset (logscale, gray = baseline, label = # significant hypotheses)', fontsize=20, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the title
    # Save
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()

# Show the testing losses of the models on a participant
def plot_testing_losses(results, dataset_name, participant, filename=None):
    plt.figure(figsize=(10, 5))
    plt.xlim((0, 100))

    # Initialize a variable to keep track of the maximum loss
    max_loss = 0
    
    palette = sns.color_palette(color_palette, len(model_names))  # Use unique colors
    
    for i, model_name in enumerate(model_names):
    
        model_training_data = results[dataset_name][participant][model_name]['model_training_data']
        baseline_training_data = results[dataset_name][participant][model_name]['baseline_training_data']
    
        # Calculate the average loss over the folds for each epoch
        model_avg_loss = np.mean(model_training_data, axis=0)
        baseline_avg_loss = np.mean(baseline_training_data, axis=0)
        
        # Update max_loss if necessary
        max_loss = max(max_loss, np.max(model_avg_loss), np.max(baseline_avg_loss))
        
        plt.plot(model_avg_loss, label=model_name, color=palette[i])
        plt.plot(baseline_avg_loss, color=palette[i], linestyle=(0, (5, 5)))
    
    # Set y-axis limits from 0 to slightly above the maximum loss
    plt.ylim(0, max_loss * 1.1)
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Testing Loss')
    plt.title(f'Average Testing Loss of Models Over {dataset_name} {participant} (dashed = baseline)', fontsize=16, fontweight='bold')
    # Adjust legend line thickness
    legend = plt.legend(loc='upper right')
    for handle in legend.legendHandles:
        handle.set_linewidth(2)  # Set the desired line width
    # Save
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()