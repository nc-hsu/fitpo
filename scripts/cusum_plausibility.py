import numpy as np
import matplotlib.pyplot as plt

def detect_cusum_change(data, threshold=4, drift=0.5):
    """
    Standard CUSUM algorithm to detect changes in a signal.
    """

    disp = data[:, 0]
    force = data[:, 1]

    velocity = np.gradient(force, disp)

    S_pos = np.zeros(len(velocity))
    S_neg = np.zeros(len(velocity))
    change_points = []
    
    mean = np.mean(velocity)
    std = np.std(velocity)
    
    # Standardize data to look for deviations from the mean
    s_data = (velocity - mean) / std if std != 0 else data - mean

    for i in range(1, len(s_data)):
        S_pos[i] = max(0, S_pos[i-1] + s_data[i] - drift)
        S_neg[i] = min(0, S_neg[i-1] + s_data[i] + drift)
        
        if S_pos[i] > threshold:
            change_points.append(('pos', i))
            S_pos[i] = 0  # Reset statistic after trigger
        elif S_neg[i] < -threshold:
            change_points.append(('neg', i))
            S_neg[i] = 0  # Reset statistic after trigger
            
    return change_points, S_pos, S_neg

def run_pushover_analysis(poc_path, threshold=4, drift=0.5):
    # Load data using numpy
    data = np.loadtxt(poc_path, delimiter=",")
    disp = data[:, 0]
    force = data[:, 1]
    
    # Calculate gradient (velocity)
    # velocity = np.gradient(force, disp)
    
    # Run CUSUM
    triggers, s_pos, s_neg = detect_cusum_change(data, threshold=threshold, drift=drift)
    
    # Identify markers for the specific drop zone near 200mm
    drops = [idx for label, idx in triggers 
                   if label == 'neg']

    rises = [idx for label, idx in triggers 
                   if label == 'pos']

    ps = np.array((disp[drops], force[drops])) if drops else None
    p_start = (disp[drops[0]-1], force[drops[0]-1]) if drops else None
    p_end = (disp[drops[-1]+1], force[drops[-1]+1]) if drops else None # end of drop

    rs = np.array((disp[rises], force[rises])) if drops else None
    r_start = (disp[rises[0]-1], force[rises[0]-1]) if drops else None
    r_end = (disp[rises[-1]+1], force[rises[-1]+1]) if drops else None # end of drop

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Top: Capacity Curve and Identified Points
    ax1.plot(disp, force, color='gray', label='Full Pushover Curve')
    if ps.any():
        ax1.scatter(*p_start, color='green', s=100, label=f'Drop Start {p_start[0]:.2f}mm', zorder=5)
        ax1.scatter(*ps, color='red', s=100, label=f'Drops', zorder=4)
        ax1.scatter(*p_end, color='blue', s=100, label=f'Drop End: {p_end[0]:.2f}mm', zorder=5)
        ax1.scatter(*r_start, color='magenta', s=100, label=f'Rise Start {r_start[0]:.2f}mm', zorder=5)
        ax1.scatter(*rs, color='yellow', s=100, label=f'Rises', zorder=4)
        ax1.scatter(*r_end, color='orange', s=100, label=f'Rises End: {r_end[0]:.2f}mm', zorder=5)
        ax1.axvspan(p_start[0], p_end[0], color='yellow', alpha=0.2, label='Capacity Drop Zone')

    ax1.set_ylabel('Force / Capacity')
    ax1.set_title('Full Curve CUSUM Analysis')
    ax1.legend()
    ax1.grid(True, linestyle=':')

    # Bottom: CUSUM Statistics (Accumulated Deviations)
    ax2.plot(disp, s_pos, label='S_pos (Increasing Slope)', color='green')
    ax2.plot(disp, s_neg, label='S_neg (Decreasing Slope)', color='red')
    ax2.axhline(4, color='black', linestyle='--', alpha=0.5, label='Threshold')
    ax2.axhline(-4, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('CUSUM Statistic')
    ax2.set_xlabel('Displacement (mm)')
    ax2.set_title('Cumulative Sum (Gradient Deviations)')
    ax2.legend()
    ax2.grid(True, linestyle=':')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from pathlib import Path
    poc_path = Path(r"C:\Users\clemettn\Documents\pypackages\fitpo\example_po_curves") / "po_curve_71.csv"
    threshold = 2
    drift = 0.3
    run_pushover_analysis(poc_path, threshold=threshold, drift=drift)