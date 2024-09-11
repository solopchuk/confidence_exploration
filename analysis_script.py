# %% imports
import os.path
import matplotlib.pyplot as plt
import numpy as np
from pingouin import corr, ttest
from scipy.stats import pearsonr
from supporting_functions import *

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# %% data loading
experiment2 = 0  # if 0 - exp1, full confidence scale, else - exp2, 0 to 50

if experiment2:
    fname = "data/data_exp2.npy"
else:
    fname = "data/data_exp1.npy"

# check if data already in folder, then just load, otherwise process the raw data
if os.path.isfile(fname):
    all_variables = np.load(fname)
else:
    generate_data_files()
    all_variables = np.load(fname)

# %% fit the model from Wilson et at '14 to choices
# H5 and H10 stand for horizon 1 and 6 in the paper respectively.
parameters_dir = "parameter_fits/"
if experiment2:
    fname = parameters_dir + "parameters_choice_exp2.npz"
else:
    fname = parameters_dir + "parameters_choice_exp1.npz"

if os.path.isfile(fname):
    data = np.load(fname)
    all_params_choice_H5 = data["all_params_choice_H5"]
    all_params_choice_H10 = data["all_params_choice_H10"]
    all_NLL_H5 = data["all_NLL_H5"]
    all_NLL_H10 = data["all_NLL_H10"]
else:
    all_params_choice_H5 = np.zeros((len(all_variables), 2))
    all_params_choice_H10 = np.zeros((len(all_variables), 2))
    all_NLL_H5, all_NLL_H10 = [np.zeros((len(all_variables), 1)) for i in range(2)]
    for sub in range(len(all_variables)):
        print(sub)
        variables = all_variables[sub]
        all_params_choice_H5[sub], all_NLL_H5[sub] = fit_logistic_model(
            variables=select_based_on_info(variables, info=1, horizon=5),
            fit_choice_immed_total=5,
        )
        all_params_choice_H10[sub], all_NLL_H10[sub] = fit_logistic_model(
            variables=select_based_on_info(variables, info=1, horizon=10),
            fit_choice_immed_total=5,
        )
    # save for later
    if not os.path.isdir(parameters_dir):
        os.mkdir(parameters_dir)
    np.savez(
        fname,
        all_params_choice_H5=all_params_choice_H5,
        all_params_choice_H10=all_params_choice_H10,
        all_NLL_H5=all_NLL_H5,
        all_NLL_H10=all_NLL_H10,
    )

# %% fit the same to confidence, but with bounds

if experiment2:
    fname = parameters_dir + "parameters_confidence_with_bounds_exp2.npz"
else:
    fname = parameters_dir + "parameters_confidence_with_bounds_exp1.npz"

if os.path.isfile(fname):
    data = np.load(fname)
    all_params_immed_H10 = data["immed_H10"]
    all_params_total_H10 = data["total_H10"]
    all_NLL_H10_conf_H10_immed = data["all_NLL_H10_conf_H10_immed"]
    all_NLL_H10_conf_H10_total = data["all_NLL_H10_conf_H10_total"]

else:
    all_params_immed_H10 = np.zeros((len(all_variables), 4))
    all_params_total_H10 = np.zeros((len(all_variables), 4))
    all_NLL_H10_conf_H10_immed = np.zeros((len(all_variables), 1))
    all_NLL_H10_conf_H10_total = np.zeros((len(all_variables), 1))

    for sub in range(len(all_variables)):
        print(sub)
        variables = all_variables[sub]
        all_params_immed_H10[sub], all_NLL_H10_conf_H10_immed[sub] = fit_brute(
            variables=select_based_on_info(variables, info=1, horizon=10),
            fit_choice_immed_total=3,
        )
        all_params_total_H10[sub], all_NLL_H10_conf_H10_total[sub] = fit_brute(
            variables=select_based_on_info(variables, info=1, horizon=10),
            fit_choice_immed_total=4,
        )

    np.savez(
        fname,
        immed_H10=all_params_immed_H10,
        total_H10=all_params_total_H10,
        all_NLL_H10_conf_H10_immed=all_NLL_H10_conf_H10_immed,
        all_NLL_H10_conf_H10_total=all_NLL_H10_conf_H10_total,
    )

# %% compute confidence variables
(
    all_means_H5_immediate_lm,
    all_means_H10_immediate_lm,
    all_means_H10_total_lm,
) = [np.zeros((len(all_variables), 2)) for i in range(3)]
diff_trials = np.empty(0)
diff_confid = []
for sub in range(len(all_variables)):
    V = all_variables[sub]
    all_confid = V["all_confid"]
    all_confid_total = V["all_confid_total"]
    all_trials = np.arange(V["all_confid"].shape[0])

    # rescale here if experiment 2 because the scale shown was 50-100 but saved as 0-100
    if experiment2:
        all_confid = 50 + 0.5 * all_confid
        all_confid_total = 50 + 0.5 * all_confid_total

    all_diff, all_chosen_uncertain, all_chosen_higher_mean = [
        np.zeros_like(V["all_choice"]) for i in range(3)
    ]

    for trial, choice in enumerate(V["all_choice"]):
        if choice == 0:
            all_diff[trial] = V["all_mean_L"][trial] - V["all_mean_R"][trial]
        else:
            all_diff[trial] = V["all_mean_R"][trial] - V["all_mean_L"][trial]
        if V["all_info_condition"][trial] == 1:
            if (V["all_presented"][trial] < 0.5 and choice == 1) or (
                V["all_presented"][trial] > 0.5 and choice == 0
            ):
                all_chosen_uncertain[trial] = 1
            else:
                all_chosen_uncertain[trial] = 0
        else:
            all_chosen_uncertain[trial] = -999  # equal information

        if ((V["all_mean_R"][trial] >= V["all_mean_L"][trial]) and choice == 1) or (
            (V["all_mean_R"][trial] < V["all_mean_L"][trial]) and choice == 0
        ):
            all_chosen_higher_mean[trial] = 1

    # H5, immediate confidence, lower mean chosen
    chosen_info_index = np.logical_and(
        np.logical_and(V["all_horizon"] == 5.0, all_chosen_uncertain == 1),
        1 - all_chosen_higher_mean,
    )
    not_chosen_info_index = np.logical_and(
        np.logical_and(V["all_horizon"] == 5.0, all_chosen_uncertain == 0),
        1 - all_chosen_higher_mean,
    )
    all_means_H5_immediate_lm[sub] = np.array(
        [
            np.mean(all_confid[chosen_info_index]),
            np.mean(all_confid[not_chosen_info_index]),
        ]
    )

    # H10, lower mean chosen
    chosen_info_index = np.logical_and(
        np.logical_and(V["all_horizon"] == 10.0, all_chosen_uncertain == 1),
        1 - all_chosen_higher_mean,
    )
    not_chosen_info_index = np.logical_and(
        np.logical_and(V["all_horizon"] == 10.0, all_chosen_uncertain == 0),
        1 - all_chosen_higher_mean,
    )
    all_means_H10_immediate_lm[sub] = np.array(
        [
            np.mean(all_confid[chosen_info_index]),
            np.mean(all_confid[not_chosen_info_index]),
        ]
    )
    all_means_H10_total_lm[sub] = np.array(
        [
            np.mean(all_confid_total[chosen_info_index]),
            np.mean(all_confid_total[not_chosen_info_index]),
        ]
    )

    diff_trials = np.append(diff_trials, all_trials[chosen_info_index])

    dconfid = all_confid_total[chosen_info_index] - all_confid[chosen_info_index]
    # diff_confid = np.append(diff_confid, dconfid - np.mean(dconfid))
    diff_confid = np.append(diff_confid, dconfid)
    print(all_confid_total[chosen_info_index] - all_confid[chosen_info_index])
    print(all_trials[chosen_info_index])

corr(diff_trials, diff_confid)

figsize = (5, 2.7)
plt.plot(diff_trials, diff_confid, ".")
plt.xlabel("trials", fontsize=12)
plt.ylabel("conf. total - conf. immediate", fontsize=12)
plt.savefig("diff_time_" + str(experiment2 + 1) + ".pdf", bbox_inches="tight", dpi=500)

# %%
for sub in range(len(all_variables)):
    plt.plot(
        all_trials[chosen_info_index],
        all_confid_total[chosen_info_index] - all_confid[chosen_info_index],
    )

# %% Figure 2 in the paper - plot actual and fitted choice probabilities and confidence judgements
# parameters fit to choices
if experiment2:
    fname = "parameter_fits/parameters_choice_exp2.npz"
else:
    fname = "parameter_fits/parameters_choice_exp1.npz"
if os.path.isfile(fname):
    data = np.load(fname)
    all_params_choice_H5 = data["all_params_choice_H5"]
    all_params_choice_H10 = data["all_params_choice_H10"]
else:
    print("loading error")
    exit(1)

# parameters fit to confidence, with bounds
if experiment2:
    fname_lapse = "parameter_fits/parameters_confidence_with_bounds_exp2.npz"
else:
    fname_lapse = "parameter_fits/parameters_confidence_with_bounds_exp1.npz"

if os.path.isfile(fname_lapse):
    data = np.load(fname_lapse)
    all_params_immed_H10 = data["immed_H10"]
    all_params_total_H10 = data["total_H10"]
else:
    print("loading error")

# grid just for plotting
N = 201
n_bins = 5
variables = np.zeros(
    N, dtype=[("all_mean_L", "<f4"), ("all_mean_R", "<f4"), ("all_presented", "<f4")]
)
variables["all_mean_L"] = np.full(N, 50.0)
variables["all_mean_R"] = np.linspace(20, 80, N)
variables["all_presented"] = np.full(N, 0.25)
# fraction of samples for right, right is uncertain

horizon_conditions = [5, 10]
info_condition = 1  # 0 equal information, 1 unequal

all_model_choice_probs = np.zeros((len(all_variables), N, n_bins))
all_model_immed_probs = np.zeros((len(all_variables), N))
all_model_total_probs = np.zeros((len(all_variables), N))
performance = np.zeros(len(all_variables))

delta_grid = np.linspace(-30, 30, n_bins)

(
    all_rates_choice,
    all_rates_confidence_immed_infochosen,
    all_rates_confidence_total_infochosen,
) = [np.ones((len(all_variables), len(delta_grid), 2)) * np.nan for i in range(3)]

informative_choice_rate = np.zeros((len(all_variables), 2))

for sub in range(len(all_variables)):

    all_model_choice_probs[sub, :, 0] = eval_logistic_model(
        variables, all_params_choice_H5[sub], fit_choice_immed_total=5
    )[:, 1]
    all_model_choice_probs[sub, :, 1] = eval_logistic_model(
        variables, all_params_choice_H10[sub], fit_choice_immed_total=5
    )[:, 1]
    all_model_immed_probs[sub, :] = eval_logistic_model(
        variables, all_params_immed_H10[sub], fit_choice_immed_total=3
    )[:, 1]
    all_model_total_probs[sub, :] = eval_logistic_model(
        variables, all_params_total_H10[sub], fit_choice_immed_total=4
    )[:, 1]

    for hh, horizon_condition in enumerate(horizon_conditions):

        V = all_variables[sub]

        ind = np.logical_and(
            V["all_horizon"] == horizon_condition,
            V["all_info_condition"] == info_condition,
        )
        V = V[ind]  # select only the variables according to the index above
        all_confid = V["all_confid"]
        all_confid_total = V["all_confid_total"]
        # rescale here if new dataset
        if experiment2:
            all_confid = 50 + 0.5 * all_confid
            all_confid_total = 50 + 0.5 * all_confid_total

        if horizon_condition == 10 and info_condition == 1:
            performance[sub] = np.mean(V["higher_chosen_rate"])

        all_delta = np.zeros(V.shape[0])
        informative = 1 - np.round(V["all_presented"])
        informative_chosen = V["all_choice"] == informative
        informative_choice_rate[sub, hh] = (
            np.sum(informative_chosen) / informative_chosen.shape[0]
        )
        repeated_next = V["all_choice_next"] == V["all_choice"]

        for trial in range(V.shape[0]):
            if informative[trial] == 1:
                all_delta[trial] = V["all_mean_R"][trial] - V["all_mean_L"][trial]
            else:
                all_delta[trial] = V["all_mean_L"][trial] - V["all_mean_R"][trial]

        # binning procedure to replicate Wilson et al figure
        grid_fid = np.diff(delta_grid)[0]
        for gg in range(len(delta_grid)):
            if gg == 0:
                index = all_delta < delta_grid[gg] + grid_fid / 2
            elif gg == len(delta_grid):
                index = all_delta >= delta_grid[gg] - grid_fid / 2
            else:
                index = np.logical_and(
                    all_delta >= delta_grid[gg] - grid_fid / 2,
                    all_delta < delta_grid[gg] + grid_fid / 2,
                )

            if np.sum(index) > 0:
                all_rates_choice[sub, gg, hh] = np.nanmean(informative_chosen[index])

            index_by_choice = np.logical_and(index, informative_chosen)
            if np.sum(index_by_choice) > 0:
                all_rates_confidence_immed_infochosen[sub, gg, hh] = np.nanmean(
                    all_confid[index_by_choice]
                )
                all_rates_confidence_total_infochosen[sub, gg, hh] = np.nanmean(
                    all_confid_total[index_by_choice]
                )

# and now plot them
plt.figure(figsize=(15, 2.7))
colors = ["#1f77b4", "#ff7f0e", "red"]
colors_confidence = ["#1f77b4", "tab:pink", "red"]

plt.subplot(1, 3, 1)
mean_choice_rate = np.nanmean(all_rates_choice, 0)
std_choice_rate = np.nanstd(all_rates_choice, 0) / np.sqrt(len(all_variables))  # SE
labels_action = ["H1", "H6"]
labels = ["H1 immediate", "H6 immediate"]
for hh in range(2):
    plt.plot(
        delta_grid, mean_choice_rate[:, hh], "o", label=labels_action[hh], c=colors[hh]
    )
    plt.errorbar(
        delta_grid,
        mean_choice_rate[:, hh],
        yerr=std_choice_rate[:, hh],
        linestyle="",
        color=colors[hh],
    )
plt.plot(
    variables["all_mean_R"] - variables["all_mean_L"],
    np.mean(all_model_choice_probs, 0)[:, 0],
    c=colors[0],
)
plt.plot(
    variables["all_mean_R"] - variables["all_mean_L"],
    np.mean(all_model_choice_probs, 0)[:, 1],
    c=colors[1],
)
plt.ylabel("p(uncertain)", fontsize=12)
plt.xlabel("mean(uncertain) - mean(known)", fontsize=12)
plt.legend(fontsize=12)
plt.axhline(y=0.5, color="gray", linestyle="--")
plt.axvline(x=0, color="gray", linestyle="--")
plt.ylim([0, 1])

plt.subplot(1, 3, 2)
mean_conf_immed_infochosen = np.nanmean(all_rates_confidence_immed_infochosen, 0)
std_conf_immed_infochosen = np.nanstd(
    all_rates_confidence_immed_infochosen, 0
) / np.sqrt(
    len(all_variables)
)  # SE
for hh in range(1, 2):  # can plot horizon 5 by changing to range(0,2)

    plt.plot(
        delta_grid,
        mean_conf_immed_infochosen[:, hh],
        "o",
        c=colors_confidence[hh],
        label=labels[hh],
    )
    plt.errorbar(
        delta_grid,
        mean_conf_immed_infochosen[:, hh],
        yerr=std_conf_immed_infochosen[:, hh],
        linestyle="",
        color=colors_confidence[hh],
    )
    # individual subjects
    for sub in range(len(all_variables)):
        plt.plot(
            delta_grid - 1 - np.abs(np.random.rand(len(delta_grid))),
            all_rates_confidence_immed_infochosen[sub, :, hh],
            ".",
            c=colors_confidence[hh],
            alpha=0.1,
        )
mean_conf_total_infochosen = np.nanmean(all_rates_confidence_total_infochosen, 0)[:, 1]
std_conf_total_infochosen = np.nanstd(all_rates_confidence_total_infochosen, 0)[
    :, 1
] / np.sqrt(len(all_variables))
plt.plot(
    delta_grid,
    mean_conf_total_infochosen,
    "o",
    c=colors_confidence[-1],
    label="H6 total",
)
plt.errorbar(
    delta_grid,
    mean_conf_total_infochosen,
    yerr=std_conf_total_infochosen,
    linestyle="",
    color=colors_confidence[-1],
)
# individual subjects
for sub in range(len(all_variables)):
    noise = np.abs(np.random.rand(len(delta_grid)))
    plt.plot(
        delta_grid + 1 + noise,
        all_rates_confidence_total_infochosen[sub, :, 1],
        ".",
        c=colors_confidence[-1],
        alpha=0.1,
    )
# model curves
# rescale here if exp2
if experiment2:
    plt.plot(
        variables["all_mean_R"] - variables["all_mean_L"],
        50 + 0.5 * np.mean(all_model_immed_probs, 0) * 100,
        c=colors_confidence[1],
    )
    plt.plot(
        variables["all_mean_R"] - variables["all_mean_L"],
        50 + 0.5 * np.mean(all_model_total_probs, 0) * 100,
        c=colors_confidence[2],
    )
else:
    plt.plot(
        variables["all_mean_R"] - variables["all_mean_L"],
        np.mean(all_model_immed_probs, 0) * 100,
        c=colors_confidence[1],
    )
    plt.plot(
        variables["all_mean_R"] - variables["all_mean_L"],
        np.mean(all_model_total_probs, 0) * 100,
        c=colors_confidence[2],
    )

plt.ylabel("confidence", fontsize=12)
plt.xlabel("mean(uncertain) - mean(known)", fontsize=12)
plt.legend(fontsize=12)
plt.title("uncertain chosen", fontsize=15)

plt.subplot(1, 3, 3)
colors = ["#1f77b4", "#ff7f0e", "red"]
colors_confidence = ["#1f77b4", "tab:pink", "red"]
plt.plot(
    np.ones((all_means_H5_immediate_lm.shape[0], 2)) * [-0.2, 0.8],
    all_means_H5_immediate_lm,
    ".",
    markersize=7,
    alpha=0.1,
    color=colors_confidence[0],
)
plt.plot(
    [-0.2, 0.8],
    np.nanmean(all_means_H5_immediate_lm, 0),
    ".",
    markersize=13,
    color=colors_confidence[0],
    label="H1",
    mfc="none",
)
plt.plot(
    np.ones((all_means_H10_immediate_lm.shape[0], 2)) * [0, 1],
    all_means_H10_immediate_lm,
    ".",
    markersize=7,
    alpha=0.1,
    color=colors_confidence[1],
)
plt.plot(
    [0, 1],
    np.nanmean(all_means_H10_immediate_lm, 0),
    ".",
    markersize=13,
    color=colors_confidence[1],
    label="H6 immediate",
    mfc="none",
)
plt.plot(
    np.ones((all_means_H10_total_lm.shape[0], 2)) * [0.2, 1.2],
    all_means_H10_total_lm,
    ".",
    markersize=7,
    alpha=0.1,
    color=colors_confidence[-1],
)
plt.plot(
    [0.2, 1.2],
    np.nanmean(all_means_H10_total_lm, 0),
    ".",
    markersize=13,
    color=colors_confidence[-1],
    label="H6 total",
    mfc="none",
)

plt.ylabel("confidence", fontsize=12)
plt.gca().set_xticks([0, 1])
plt.gca().set_xticklabels(["Uncertain chosen", "Known chosen"], fontsize=12)
plt.title("lower mean chosen", fontsize=15)
plt.legend()
plt.show()
# plt.savefig('qualitative_' + str(experiment2+1) + '.pdf', bbox_inches="tight", dpi=500)

# %% Statistical analyses

# confidence main effect t-test
difference = all_means_H10_total_lm[:, 0] - all_means_H10_immediate_lm[:, 0]
ttest(difference[~np.isnan(difference)], y=0)

# test the difference in parameters
# information seeking choice
difference = all_params_choice_H10[:, 0] - all_params_choice_H5[:, 0]
ttest(difference[~np.isnan(difference)], y=0)

# temperature choice
difference = all_params_choice_H10[:, 1] - all_params_choice_H5[:, 1]
ttest(difference[~np.isnan(difference)], y=0)

# information seeking confidence
difference = all_params_total_H10[:, 0] - all_params_immed_H10[:, 0]
ttest(difference[~np.isnan(difference)], y=0)

# temperature confidences
difference = all_params_total_H10[:, 1] - all_params_immed_H10[:, 1]
ttest(difference[~np.isnan(difference)], y=0)

# choice and confidence difference - information seeking
difference = (all_params_choice_H10[:, 0] + all_params_choice_H5[:, 0]) / 2 - (
    all_params_immed_H10[:, 0] + all_params_total_H10[:, 0]
) / 2
ttest(difference[~np.isnan(difference)], y=0)

# choice and confidence difference - temperature
difference = (all_params_choice_H10[:, 1] + all_params_choice_H5[:, 1]) / 2 - (
    all_params_immed_H10[:, 1] + all_params_total_H10[:, 1]
) / 2
ttest(difference[~np.isnan(difference)], y=0)

# correlations, confidence and performance
choice_effect = all_params_choice_H10[:, 0] - all_params_choice_H5[:, 0]
confidence_effect = all_means_H10_total_lm[:, 0] - all_means_H10_immediate_lm[:, 0]

pearsonr(
    choice_effect[~np.isnan(confidence_effect)],
    confidence_effect[~np.isnan(confidence_effect)],
)
corr(
    choice_effect[~np.isnan(confidence_effect)],
    confidence_effect[~np.isnan(confidence_effect)],
)
corr(
    performance[~np.isnan(confidence_effect)],
    confidence_effect[~np.isnan(confidence_effect)],
)

# %% make matlab table for anova
table = np.zeros((len(all_variables), 8))

for sub in range(len(all_variables)):
    variables = all_variables[sub]
    variables = variables[variables["all_horizon"] == 10.0]
    variables = variables[variables["all_info_condition"] == 1.0]

    # 000, 001, 010, 011, 100, 101, 110, 111

    unc_chosen = variables["all_chosen_uncertain"]
    lm_chosen = 1 - variables["all_chosen_higher_mean"]
    confid = variables["all_confid"]
    confid_total = variables["all_confid_total"]

    table[sub, 0] = confid[np.logical_and(unc_chosen == 0, lm_chosen == 0)].mean()
    table[sub, 1] = confid[np.logical_and(unc_chosen == 0, lm_chosen == 1)].mean()
    table[sub, 2] = confid[np.logical_and(unc_chosen == 1, lm_chosen == 0)].mean()
    table[sub, 3] = confid[np.logical_and(unc_chosen == 1, lm_chosen == 1)].mean()

    table[sub, 4] = confid_total[np.logical_and(unc_chosen == 0, lm_chosen == 0)].mean()
    table[sub, 5] = confid_total[np.logical_and(unc_chosen == 0, lm_chosen == 1)].mean()
    table[sub, 6] = confid_total[np.logical_and(unc_chosen == 1, lm_chosen == 0)].mean()
    table[sub, 7] = confid_total[np.logical_and(unc_chosen == 1, lm_chosen == 1)].mean()

np.sum(np.sum(np.isnan(table), axis=1) > 0)
# pd.DataFrame(table).to_csv('table' + str(experiment2 + 1) + '.csv')

# %% Figure 3 in the paper, plot model parameters, Experiment labels and significance stars added in post-processing

plt.figure(figsize=(15, 2.7))
for experiment2 in range(2):
    if experiment2:
        fname = "parameter_fits/parameters_choice_exp2.npz"
    else:
        fname = "parameter_fits/parameters_choice_exp1.npz"
    if os.path.isfile(fname):
        data = np.load(fname)
        all_params_choice_H5 = data["all_params_choice_H5"]
        all_params_choice_H10 = data["all_params_choice_H10"]
    else:
        print("loading error")
        exit(1)

    # confidence
    if experiment2:
        fname = "parameter_fits/parameters_confidence_with_bounds_exp2.npz"
    else:
        fname = "parameter_fits/parameters_confidence_with_bounds_exp1.npz"
    if os.path.isfile(fname):
        data = np.load(fname)
        all_params_immed_H10 = data["immed_H10"]
        all_params_total_H10 = data["total_H10"]
    else:
        print("loading error")
        exit(1)

    colors_fw_bw_plot = ["#1f77b4", "#ff7f0e", "tab:pink", "red"]
    for param, title in enumerate(["information bias", "temperature"]):
        plt.subplot(
            1, 4, experiment2 * 2 + param + 1
        )  # 4 subplots is just a trick to get the right size
        plt.plot(
            np.ones_like(all_params_choice_H5[:, param]),
            all_params_choice_H5[:, param],
            ".",
            markersize=7,
            alpha=0.1,
            color=colors_fw_bw_plot[0],
        )
        plt.plot(
            2 * np.ones_like(all_params_choice_H10[:, param]),
            all_params_choice_H10[:, param],
            ".",
            markersize=7,
            alpha=0.1,
            color=colors_fw_bw_plot[1],
        )
        plt.plot(
            3 * np.ones_like(all_params_choice_H10[:, param]),
            all_params_immed_H10[:, param],
            ".",
            markersize=7,
            alpha=0.1,
            color=colors_fw_bw_plot[2],
        )
        plt.plot(
            4 * np.ones_like(all_params_choice_H10[:, param]),
            all_params_total_H10[:, param],
            ".",
            markersize=7,
            alpha=0.1,
            color=colors_fw_bw_plot[3],
        )
        plt.plot(
            [1, 2],
            [
                np.mean(all_params_choice_H5[:, param], 0),
                np.mean(all_params_choice_H10[:, param], 0),
            ],
            ".",
            markersize=10,
            color="k",
            linestyle="dashed",
            mfc="none",
        )
        plt.plot(
            [3, 4],
            [
                np.mean(all_params_immed_H10[:, param], 0),
                np.mean(all_params_total_H10[:, param], 0),
            ],
            ".",
            markersize=10,
            color="k",
            linestyle="dashed",
            mfc="none",
        )
        plt.title(title, fontsize=15)
        plt.gca().set_xticks([1, 2, 3, 4])
        plt.gca().set_xticklabels(
            ["choice \nH1", "choice \nH6", "conf \nimm. \nH6", "conf \ntotal \nH6"],
            fontsize=15,
        )

# plt.savefig('params_' + str(experiment2+1) + '.pdf', bbox_inches="tight", dpi=500)

# %% Supplementary figure 1 - check confidence calibration for different scales

nBins = 10
all_colors = ["y", "c"]
plt.figure(figsize=(4, 4))
for experiment2 in range(2):

    if experiment2:
        fname = "data/data_exp2.npy"
    else:
        fname = "data/data_exp1.npy"

    # check if data already in folder, then just load, otherwise process
    if os.path.isfile(fname):
        all_variables = np.load(fname)
    else:
        print("loading error")
        exit(1)

    if experiment2:
        LB = 50
    else:
        LB = 0

    threshs = np.linspace(LB, 100, nBins + 1)
    allCA = np.zeros((len(all_variables), nBins))
    allCwW = np.zeros((len(all_variables), 1))
    for sub in range(len(all_variables)):
        V = all_variables[sub]
        # V = V[V['all_horizon']==5]
        V = V[V["all_info_condition"] == 0]
        all_confid = V["all_confid"]
        if experiment2:
            all_confid = 50 + 0.5 * all_confid

        for bin in range(len(threshs) - 1):
            allCA[sub, bin] = get_choosing_higher_mean_rate(
                V[
                    np.logical_and(
                        all_confid > threshs[bin], all_confid < threshs[bin + 1]
                    )
                ]
            )

        all_chosen_higher_mean = np.zeros_like(V["all_choice"])
        for trial, choice in enumerate(V["all_choice"]):
            if ((V["all_mean_R"][trial] > V["all_mean_L"][trial]) and choice == 1) or (
                (V["all_mean_R"][trial] < V["all_mean_L"][trial]) and choice == 0
            ):
                all_chosen_higher_mean[trial] = 1
        allCwW[sub] = np.mean(all_confid[all_chosen_higher_mean == 0])

    MA = np.zeros(len(threshs) - 1)
    for bin in range(len(threshs) - 1):
        MA[bin] = (threshs[bin] + threshs[bin + 1]) / 2

    if experiment2 == 0:
        plt.plot([0, 100], [0, 100], "k--", alpha=0.4)
    MN = np.nanmean(allCA, 0) * 100
    SE = np.nanstd(allCA, 0) / np.sqrt(allCA.shape[0]) * 100

    for bin in range(len(MA)):
        SE_bin = (
            np.nanstd(allCA[:, bin], 0)
            / np.sqrt(np.sum(~np.isnan(allCA[:, bin])))
            * 100
        )
        plt.plot(
            [MA[bin], MA[bin]], [MN[bin], MN[bin] + SE_bin], markersize=10, c="gray"
        )
        plt.plot(
            [MA[bin], MA[bin]], [MN[bin], MN[bin] - SE_bin], markersize=10, c="gray"
        )
        if bin == 0:
            plt.plot(
                MA[bin],
                MN[bin],
                ".",
                label="exp " + str(experiment2 + 1),
                markersize=8,
                c=all_colors[experiment2],
            )
        else:
            plt.plot(MA[bin], MN[bin], ".", markersize=8, c=all_colors[experiment2])


plt.xlabel("confidence (immediate reward)", fontsize=12)
plt.ylabel("choosing higher mean, %", fontsize=12)
plt.legend()
plt.gca().spines[["right", "top"]].set_visible(False)
plt.show()
# plt.tight_layout()
# plt.savefig('calibration.pdf', bbox_inches="tight", dpi=500)
