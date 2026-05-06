
# Build a mutual-fund policy slice and visualize x_t and x_mt against P_t.
model=MutualFundModel()
model.config.Ee = (0.0, 0.0, 0.0)
model.config.var_cov = (
    (0.030186, 0.0, 0.0),
    (0.0, 0.017292, 0.0),
    (0.0, 0.0, 0.026941),
)
copt = solve_mutual_coefficients(model.config)
policy = mutual_policy_grid(copt, model.config, t=model.config.T - 1, Lt=600.0, Rt=365.0)

fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
wealth_labels = [f'W={int(w/1000)}k' for w in policy['W']]

for wi, label in enumerate(wealth_labels):
    axes[0].plot(policy['P'], policy['X'][:, wi], label=label)
    axes[1].plot(policy['P'], policy['XM'][:, wi], label=label)

axes[0].set_title('Optimal land adjustment x_t')
axes[1].set_title('Optimal mutual-fund investment x_mt')
for ax in axes:
    ax.set_xlabel('P_t')
    ax.set_ylabel('Control')
    ax.grid(True, alpha=0.3)
axes[0].legend(loc='best', fontsize=8)
fig.tight_layout()
plt.show()