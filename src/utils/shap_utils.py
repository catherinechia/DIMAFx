import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def shap_dotplot_by_risk(shap_vals, feature_names, risk_scores, cmap="coolwarm", save_path=None, log2=True, cbar_title='Log2\nSurvival\nRisk'):
    """ Make shap risk plot. Set log2=False to color by scores that are already centered at 0 (e.g. log-odds). """
    if log2:
        risk_scores = np.log2(risk_scores)

    # Normalize colors tightly around your risk scores
    # vmin, vmax = np.min(risk_scores), np.max(risk_scores)
    # norm = plt.Normalize(vmin=vmin, vmax=vmax)
    max_val = np.max(np.abs(risk_scores))
    norm = TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
    cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(12, len(feature_names) * 0.4))

    features = zip(range(len(feature_names)), feature_names, np.mean(np.abs(shap_vals), axis=0))
    # order features by mean absolute SHAP value
    features = sorted(features, key=lambda x: x[2], reverse=False)
    
    for i, (pos, feat_name, abs_shap) in enumerate(features):
        sc = ax.scatter(
            shap_vals[:, pos],                  # x = SHAP values
            np.full(len(shap_vals), i),       # y = feature position
            c=risk_scores,                      # color by risk
            cmap=cmap,
            norm=norm,                          # <- custom normalization
            s=20,
            alpha=0.8,
            edgecolor="none"
        )

    # Formatting
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([f[1] for f in features])
    ax.axvline(0, color="k", linestyle="-", linewidth=1)
    ax.set_ylabel("")
    sns.despine(ax=ax, top=True, right=True, left=True)
    ax.set_xlabel("SHAP value")

    # Colorbar
    cbar = plt.colorbar(sc, orientation='vertical', shrink=0.3, aspect=10, pad=0.08)
    cbar.ax.set_title(cbar_title, fontsize=10, pad=4)

    ax.set_ylim(-0.5, len(feature_names) - 0.5)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.show()



def find_feat_name(ft_name):
    """ Find correct name under which the feature is stored. """
    feat_name_splitted = re.match(r'([A-Za-z]+)(\d+)', ft_name).groups() 
    if feat_name_splitted[0] == "R":
        name = f'rna_pt_{feat_name_splitted[1]}'
    else:
        name = f'wsi_pt_{feat_name_splitted[1]}'

    return name

def get_vals_feature(shap_dict, feat_name):
    """ Get the shap value for the interaction plot. """
    feat_name_shap_dict = find_feat_name(feat_name)
    index = shap_dict['Feature names'].index(feat_name_shap_dict)
    values = shap_dict['shap values'][:, index, :]
    feat_vals = np.sum(values, axis=1)
    return feat_vals


def get_vals(values, axis=(1,2)):
    """ Get mean absolute SHAP values. """
    sum_vals = np.sum(values, axis=axis)
    shap_val = np.mean(np.absolute(sum_vals), axis=(0))
    return shap_val

def create_dataframe(spec, shar, feat_names):
    """Create a DataFrame from SHAP values."""
    df = pd.DataFrame({
        "Modality Specific": spec,
        "Modality Shared": shar,
    }, index=feat_names)
    return df

def plot_shaps_comparison(df, label_features=[], xlim=None, ylim=None, colors=None, with_legend=True):
    """ Plot the comparison between modality specific and modality shared features. """
    # Assign a unique color to each feature
    if colors is None:
        colors = sns.color_palette("husl", len(df)) 

    # Scatter plot
    plt.figure()
    for feature, color in zip(df.index, colors):
        plt.scatter(df.loc[feature, "Modality Shared"], df.loc[feature, "Modality Specific"],
                    color=color, label=feature, s=25)
        short_name = feature.split(" ")[0][:-1]
        if short_name in label_features:
            plt.text(df.loc[feature, "Modality Shared"], df.loc[feature, "Modality Specific"], short_name, fontsize=7, ha='right', va='bottom')

    # Plot 45-degree line (y = x)
    lims = [
        min(df["Modality Shared"].min(), df["Modality Specific"].min()),
        max(df["Modality Shared"].max(), df["Modality Specific"].max())
    ]
    plt.plot(lims, lims, 'k--', label="y = x")  # black dashed line
    plt.xlim(0)
    plt.ylim(0)
    if xlim:
        plt.xlim((0, xlim))

    if ylim:
        plt.ylim((0, ylim))

    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("SHAP of modality shared features", fontsize=11)
    plt.ylabel("SHAP of modality\nspecific features", fontsize=11)
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=6)
    if len(colors) < 26:
        ncol=1
    elif len(colors) < 51:
        ncol=2
    else:
        ncol=3

    if with_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2, title="Features")
    
    plt.show()

