import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

df = pd.read_csv('data/Game.csv')

# Get article_open events for real trials (IsPractice is NaN)
opens = df[(df['Action'] == 'article_open') & (df['IsPractice'].isna())].copy()
opens['Time'] = pd.to_datetime(opens['Time'])
opens = opens.sort_values(['ID', 'TrialIndex', 'Time'])

# Page sequence number within each trial
opens['PageNum'] = opens.groupby(['ID', 'TrialIndex']).cumcount() + 1

# Get task_end times to cap the last page
task_ends = df[df['Action'] == 'task_end'][['ID', 'TrialIndex', 'Time']].copy()
task_ends['Time'] = pd.to_datetime(task_ends['Time'])
task_ends = task_ends.rename(columns={'Time': 'TaskEndTime'})
opens = opens.merge(task_ends, on=['ID', 'TrialIndex'], how='left')

# Duration = time until next page open (or task_end for last page)
opens['NextOpenTime'] = opens.groupby(['ID', 'TrialIndex'])['Time'].shift(-1)
opens['EndTime'] = opens['NextOpenTime'].fillna(opens['TaskEndTime'])
opens['Duration'] = (opens['EndTime'] - opens['Time']).dt.total_seconds()
opens = opens.dropna(subset=['Duration'])

# --- Subplots: one per user ---
users = sorted(opens['ID'].unique())
n_users = len(users)
cols = 4
rows = int(np.ceil(n_users / cols))

fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3.2), sharex=False, sharey=True)
fig.patch.set_facecolor('#0d1117')
fig.suptitle('M10 — Time Spent per Page, by User', fontsize=18, color='#e6edf3', fontweight='bold', y=0.98)

palette = plt.cm.Set2(np.linspace(0, 1, n_users))
axes_flat = axes.flatten()

for i, user_id in enumerate(users):
    ax = axes_flat[i]
    ax.set_facecolor('#0d1117')

    user_data = opens[opens['ID'] == user_id].sort_values('PageNum')

    # Separate trials with different line styles
    trials = sorted(user_data['TrialIndex'].unique())
    trial_markers = ['o', 's', 'D', '^', 'v']

    for j, trial in enumerate(trials):
        td = user_data[user_data['TrialIndex'] == trial]
        marker = trial_markers[j % len(trial_markers)]
        ax.plot(
            td['PageNum'], td['Duration'],
            marker=marker, markersize=6,
            linewidth=1.5, alpha=0.85,
            color=palette[i],
            markeredgecolor='white', markeredgewidth=0.5,
            label=f'Trial {trial}',
            zorder=3,
        )

    ax.set_title(f'User {user_id}', fontsize=11, color='#e6edf3', fontweight='bold', pad=6)
    ax.set_xlabel('Page #', fontsize=9, color='#8b949e')
    ax.set_ylabel('Time (s)', fontsize=9, color='#8b949e')
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(colors='#8b949e', labelsize=8)
    ax.tick_params(axis='y', which='minor', colors='#8b949e', length=3)
    ax.grid(True, color='#21262d', linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    max_page = int(user_data['PageNum'].max())
    ax.set_xticks(range(1, max_page + 1))

    if len(trials) > 1:
        leg = ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

# Hide unused subplots
for k in range(n_users, len(axes_flat)):
    axes_flat[k].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('output/m10_time_per_page.png', dpi=180, facecolor=fig.get_facecolor())
plt.close()
print('Saved: output/m10_time_per_page.png')
