import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import brute
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def get_sub_IDs(experiment2):
    
    # Experiment 2, confidence scale from 50 to 100
    if experiment2:
        all_sub_IDs = ['6504688c4ffe3',
                    '650468b690522',
                    '6504690d3b99f',
                    '6504718a47d3f',
                    '6504730331f74',
                    '6504738f6dcda',
                    '6504745606526',
                    '6504751c9f190',
                    '650475605d3f5',
                    '6504756c0a3dd',
                    '650475c33cdef',
                    '650475cd08c6a',
                    '6504773872892',
                    '6504780d57795',
                    '650479e687268',
                    '65047aac10b79',
                    '65047b37f1728',
                    '65047c5bcc3a2',
                    '65047c792f352',
                    '65047e0be7628',
                    '65047ec34f3b6',
                    '65048b4a3ecdf',
                    '65048b59eae71',
                    '65048be9f07a1',
                    '65048c7c2c50f',
                    '65048d49c4dd0',
                    '65048dcddc708',
                    '65048dfeeb837',
                    '65048e2204f44',
                    '65048ec78e830',
                    '65048fe20c83d',
                    '6504906d23047',
                    '650490aca8c61',
                    '650490dcadda0',
                    '650493570495a',
                    '6504936e80b00',
                    '6504950be9d9c',
                    '6504953274074',
                    '6504958fc054a',
                    '6504976f16f42',
                    '65049cba8bde7',
                    '65049cededa94',
                    '65049d8c4dbdb',
                    '6504c1e8ed8db',
                    '6504c4cca11bc',
                    '6504c538c4449',
                    '6504c5c7a3aa6',
                    '6504c61838421',
                    '6504c65e9702b',
                    '6504c98ba4d85',
                    '6504d8b57763d',
                    '6504dcd36c372',
                    '6504ddc277080',
                    '6504de69137a1',
                    '6504e1986dc62',
                    '6504f3b6c783a',
                    '6505793dd3b43',
                    '6505797456700',
                    '6505798e398b3',
                    '650579a958bbf',
                    '650579e339330',
                    '650579f08c263',
                    '65057a08256b5',
                    '65057aca2d0b8',
                    '65057ad28719c',
                    '65057b1d48577',
                    '65057b48742e9',
                    '65057b5c715a0',
                    '65057b98736f3',
                    '65057bc71c35f',
                    '65057be25ed6d',
                    '65057c354f1f1',
                    '65057d960e6f2',
                    '65057e3c6df75',
                    '65057ef0b6c2d',
                    '65057f52bc98f',
                    '650580140ac0d',
                    '6505802128479',
                    '6505827a129e3',
                    '650588cc2731b',
                    '65058b3a83241',
                    '65059363eabd1',
                    '650596c0cd321',
                    '650598fe89167',
                    '650599018dbba',
                    '6505c2972cb10',
                    '6505c29d6fab0',
                    '6505c2bd8032c',
                    '6505c3729a2fa',
                    '6505c3a14af7f',
                    '6505c3fee11aa',
                    '6505c42aa1a35',
                    '6505c4e65bf32',
                    '6505c5c469f49',
                    '6505c6089f6f9',
                    '6505c6582ec4b',
                    '6505c6a20440e',
                    '6505c6ac923a6',
                    '6505c7609834d',
                    '6505c778220e5',
                    '6505c78973151',
                    '6505c7bb1ab87',
                    '6505c7d1bfd32',
                    '6505c8fe6c7a5',
                    '6505ca1f63286',
                    '6505ca270f140',
                    '6505ca3190b4f',
                    '6505cd6d64882',
                    '6505d5fa66342']
    else:

        # Experiment 1, full confidence scale from 0 to 100
        all_sub_IDs = ['6488b567529eb',
                    '6489d0ee5204c',
                    '6489d1843fc10',
                    '6489d34bb0654',
                    '6489d3682b851',
                    '6489d9a319266',
                    '6489ef65ee16a',
                    '6489f03094c43',
                    '6489f13c4bf93',
                    '6489f2518c3ad',
                    '6489f28d4f34d',
                    '6489f9c07b3a4',
                    '648af09222284',
                    '648af43cc534e',
                    '648af72c5b999',
                    '648af7e70121c',
                    '648b17e2edc14',
                    '648b20291501c',
                    '648b243a1bcae',
                    '648b304182d88',
                    '648b31a1b54f2',
                    '648b43b16c8f5',
                    '648b4cc5dd166',
                    '648b4d410349e',
                    '648b4d7a6bfed',
                    '648b4dc83f378',
                    '648b4ef66dea0',
                    '648b4fbdbd1c5',
                    '648c35390de91',
                    '648c36dbbf36e',
                    '648c36f556ae7',
                    '648c38bebe872',
                    '648c3ee612c6a',
                    '648c54380b89d',
                    '648c54825aa6f',
                    '648c5485a6f57',
                    '648c5536399fb',
                    '648c570693611',
                    '648c6e3002c06',
                    '648c71d371c7f',
                    '648c7d56437ca',
                    '648d55d7e4502',
                    '648d55e43a972',
                    '648d567817d41',
                    '648d5678db6d2',
                    '648d58ffc4ade',
                    '648d5959e6d8b',
                    '648d59a87df22',
                    '648d5a224ed77',
                    '648d5a9d4b0db',
                    '648d5c85157c1',
                    '648d681945b0c',
                    '648d6831f2df9',
                    '648d6a8e47595',
                    '648d6afb0bc19',
                    '648d6bb78512e',
                    '648d6c2b0d7d4',
                    '648d6d8f74b53',
                    '648d6e8fa1fdd',
                    '648d7195f2bc8',
                    '648d72dfe1e3b',
                    '648f640c92581',
                    '648f644d52932',
                    '64900dae1b57b',
                    '649069ff5e4d8',
                    '64a6a2e1730cb',
                    '64a6a4b519495',
                    '64a6a4d2d9099',
                    '64a6a5099e0ee',
                    '64a6a615279c9',
                    '64a6a6c07e2e7',
                    '64a6a73c41f8c',
                    '64a6a76bde7a7',
                    '64a6a7fa50ea5',
                    '64a6ac86abc9a',
                    '64a6ac8c1b329',
                    '64a6ccd897f0a',
                    '64a6cd26358a0',
                    '64a6cd96ad6b8',
                    '64a6cdaf6c9fd',
                    '64a6ce110db94',
                    '64a6ced49e0d6',
                    '64a6d00719806',
                    '64a6d28d96cc5',
                    '64a6d47f75f30',
                    '64a6d4f8b38c6',
                    '64a6d7f2565d7',
                    '64a6e396d86c4',
                    '64a6e8f4131ea',
                    '64b2653def0be',
                    '64b267e20cb8e',
                    '64b26849aedbb',
                    '64b268730b639',
                    '64b269dd0f84d',
                    '64b269fa41575',
                    '64b26a40a14d6',
                    '64b26c2956f5e',
                    '64b26fc0cb566',
                    '64b2750d0fd6f',
                    '64b3ddbd2ab7f',
                    '64b3de85e315e',
                    '64b3dfaf3ea1f',
                    '64b3e15860a9d',
                    '64b3e39f719de',
                    '64b3ea600a426',
                    '64b3fcf352d2e',
                    '64b406ecf0ab9',
                    '64b42f020f368',
                    '64b45bc33d0a9',
                    '64b45c55b11c1',
                    '64b45d567bc81',
                    '64b45de56dd2a',
                    '64b45ded6dd8a',
                    '64b45e817f1f5',
                    '64b45eaf1e809',
                    '64b45ec2b3fdb',
                    '64b45ff96c144',
                    '64b4617d8ca31',
                    '64b461c4ee34c',
                    '64b4624d5345b',
                    '64b4634bcd92b',
                    '64b4679be000e',
                    '653939923e549',
                    '653939ab40215',
                    '65393a43c4eda',
                    '65393af484adf',
                    '65393b193f8fc',
                    '65393b319eb33',
                    '65393bdbc9d03',
                    '65393c09826cd',
                    '65393c4d340bb',
                    '65393c7656857',
                    '65393db063fd5',
                    '65393e212d208',
                    '65393f6c36afb',
                    '653941b9017f5',
                    '653961b3bd9a6',
                    '653a3cde7a596',
                    '653a491a9d78f',
                    '653a497e5b61d',
                    '653a4994251fd',
                    '653a4b0a4aedf',
                    '653a4b7bf1962',
                    '653a4b812007d',
                    '653a4bc5aa6aa',
                    '653a4cdaaeed4',
                    '653a4d18a3d98',
                    '653a4dd29e85b',
                    '653a4e6095ac6',
                    '653a5109f3c2a',
                    '653a52c25ca42',
                    '653a557d3c371',
                    '653a5ac91b39f',
                    '653a73d6ef600',
                    '653a7417f2303',
                    '653a74d3ecfed',
                    '653a8e3a81c24',
                    '653bca6cf21c7',
                    '653bcb2348fba',
                    '653bcb2c27a6b',
                    '653bcd0148ccf',
                    '653bcd43ef439',
                    '653bce05a0f47',
                    '653bce930d804',
                    '653bd1104254f',
                    '653bd2c00ad37',
                    '653be1753d288',
                    '653c15203b2a2',
                    '653c16244941a',
                    '653c165d35dfa',
                    '653c169b230df',
                    '653c16c514229',
                    '653c16f728d74',
                    '653c173b10f9c',
                    '653c174888b45',
                    '653c19dd132d6',
                    '653c1fb18eecf',
                    '653c2dd9e76fc',
                    '653c31383940e']
    return all_sub_IDs

def load_data(id):
    data = pd.read_csv(f'data/session-{id}.csv')
    return data

def clean_data(data):
    data = data.loc[(data['trial_type'] == 'horizons-trial')
                    & (data['phase'] != 'practice')]
    return data

def show_comments(data):
    # data.loc[(data['trial_type'] == 'survey-html-form')]['response']
    surveys = data.loc[(data['trial_type'] == 'survey-html-form')]
    for i in range(len(surveys)):
        print(surveys['response'].iloc[i])

def show_confidence_hist(data):
    plt.figure()
    all_conf = data['confidence'].str.extract('(\d+)').astype(int).to_numpy()
    all_conf = np.squeeze(all_conf)
    plt.hist(all_conf, color='r')
    plt.xlabel('confidence', fontsize=12)
    plt.ylabel('frequency', fontsize=12)
    plt.title('Histogram of confidence judgements', fontsize=12)

def extract_variables(data):

    variables = np.zeros(len(data), dtype=[('all_sd_unchosen', '<f4'),
                                           ('all_sd_chosen', '<f4'),
                                           ('all_mean_L', '<f4'),
                                           ('all_mean_R', '<f4'),
                                           ('all_mean_L_gen', '<f4'),
                                           ('all_mean_R_gen', '<f4'),
                                           ('all_mean_L_5turns', '<f4'),
                                           ('all_mean_R_5turns', '<f4'),
                                           ('all_choice', '<i4'),
                                           ('all_confid', '<f4'),
                                           ('all_confid_total', '<f4'),
                                           ('all_cnfind', '<i4'),
                                           ('all_confidence_RT', '<f4'),
                                           ('all_choice_RT', '<f4'),
                                           ('all_horizon', '<i4'),
                                           ('all_info_condition', '<i4'),
                                           ('all_presented', '<f4'),
                                           ('all_presented_5turns', '<f4'),
                                           ('all_choice_next', '<f4'),
                                           ('all_rew_turn5', '<f4'),
                                           ('all_belief_mean_immediate_L', '<f4'),
                                           ('all_belief_certainty_immediate_L', '<f4'),
                                           ('all_belief_mean_immediate_R', '<f4'),
                                           ('all_belief_certainty_immediate_R', '<f4'),
                                           ('all_belief_mean_total_L', '<f4'),
                                           ('all_belief_certainty_total_L', '<f4'),
                                           ('all_belief_mean_total_R', '<f4'),
                                           ('all_belief_certainty_total_R', '<f4'),
                                           ('V_chosen', '<f4'),
                                           ('V_unchosen', '<f4'),
                                           ('all_diff', '<f4'),
                                           ('all_chosen_uncertain', '<f4'),
                                           ('all_chosen_higher_mean', '<f4'),
                                           ('orientation', '<i4'),
                                           ('higher_chosen_rate', '<f4')])

    for trial in range(len(data)):

        n_choices = data['key_presses'].iloc[trial:trial +
                                             1].str.split(',', expand=True).shape[1]
        all_turn_choices = np.zeros(n_choices-4)
        for tt, turn in enumerate(range(4, n_choices)):
            selected = data['key_presses'].iloc[trial:trial+1].str.split(',', expand=True)[
                turn].str.replace(r']', '', regex=True).str.replace(r'[', '', regex=True).str.replace(r'"', '', regex=True)
            if selected.values[0] == 'arrowleft':
                choice = 0
            elif selected.values[0] == 'arrowright':
                choice = 1
            all_turn_choices[tt] = choice

        higher_mean = np.argmax([np.squeeze(
            data['r_mean_L'].iloc[trial:trial+1]), np.squeeze(data['r_mean_R'].iloc[trial:trial+1])])
        higher_chosen_rate = np.sum(
            all_turn_choices == higher_mean)/len(all_turn_choices)
        variables['higher_chosen_rate'][trial] = higher_chosen_rate

        selected = data['key_presses'].iloc[trial:trial+1].str.split(',', expand=True)[
            4].str.replace(r']', '', regex=True).str.replace(r'[', '', regex=True).str.replace(r'"', '', regex=True)
        if selected.values[0] == 'arrowleft':
            choice = 0
        elif selected.values[0] == 'arrowright':
            choice = 1

        forced_choices = np.squeeze(
            data['forced_choices'].iloc[trial:trial+1].str.extractall('(\d+)').unstack().astype(int).to_numpy())

        forced_and_first = np.append(forced_choices, choice)
        first_four_R = np.squeeze(data['rewards_R'].iloc[trial:trial+1].str.extractall(
            '(\d+)').unstack().astype(int).to_numpy())[:4]
        first_four_L = np.squeeze(data['rewards_L'].iloc[trial:trial+1].str.extractall(
            '(\d+)').unstack().astype(int).to_numpy())[:4]
        first_five_R = np.squeeze(data['rewards_R'].iloc[trial:trial+1].str.extractall(
            '(\d+)').unstack().astype(int).to_numpy())[:5]
        first_five_L = np.squeeze(data['rewards_L'].iloc[trial:trial+1].str.extractall(
            '(\d+)').unstack().astype(int).to_numpy())[:5]

        reward_turn5 = np.squeeze(data['rewards_L'].iloc[trial:trial+1].str.extractall(
            '(\d+)').unstack().astype(int).to_numpy())[4:5]
        variables['all_mean_L'][trial] = np.mean(
            first_four_L[forced_choices == 0])
        variables['all_mean_R'][trial] = np.mean(
            first_four_R[forced_choices == 1])
        variables['all_mean_L_5turns'][trial] = np.mean(
            first_five_L[forced_and_first == 0])
        variables['all_mean_R_5turns'][trial] = np.mean(
            first_five_R[forced_and_first == 1])

        RT = np.squeeze(data['response_times'].iloc[trial:trial +
                        1].str.extractall('(\d+)').unstack().astype(int).to_numpy())[4:5]
        variables['all_choice_RT'][trial] = RT
        
        if np.squeeze(data['confidence_rt'].iloc[trial:trial+1]).strip('[]') != '':
            confidence_RT = np.squeeze(data['confidence_rt'].iloc[trial:trial+1].str.extractall('(\d+)').unstack().astype(int).to_numpy())
            variables['all_confidence_RT'][trial] = confidence_RT

        if int(data['horizon'].iloc[trial:trial+1]) == 10:
            selected_next = data['key_presses'].iloc[trial:trial+1].str.split(',', expand=True)[
                5].str.replace(r']', '', regex=True).str.replace(r'[', '', regex=True).str.replace(r'"', '', regex=True)
            if selected_next.values[0] == 'arrowleft':
                choice_next = 0
            elif selected_next.values[0] == 'arrowright':
                choice_next = 1
        else:
            choice_next = np.nan

        variables['all_choice'][trial] = choice
        variables['all_choice_next'][trial] = choice_next
        variables['all_presented'][trial] = np.mean(forced_choices)
        variables['all_presented_5turns'][trial] = np.mean(forced_and_first)
        variables['all_rew_turn5'][trial] = reward_turn5
        variables['all_cnfind'][trial] = np.squeeze(
            data['confidence_immediate_overall'].iloc[trial:trial+1].astype(int).to_numpy())
        variables['all_horizon'][trial] = int(
            data['horizon'].iloc[trial:trial+1])
        variables['all_info_condition'][trial] = int(
            data['r_info'].iloc[trial:trial+1])

        if int(data['r_info'].iloc[trial:trial+1]) == 1:  # unequal
            if choice == 1:
                variables['all_sd_chosen'][trial] = np.std(
                    first_four_L[forced_choices == 1])
                variables['all_sd_unchosen'][trial] = np.std(
                    first_four_L[forced_choices == 0])

            else:
                variables['all_sd_chosen'][trial] = np.std(
                    first_four_L[forced_choices == 0])
                variables['all_sd_unchosen'][trial] = np.std(
                    first_four_R[forced_choices == 1])
        else:
            variables['all_sd_unchosen'][trial] = np.nan

        if choice == 0:
            variables['V_chosen'][trial] = variables['all_mean_L'][trial]
            variables['V_unchosen'][trial] = variables['all_mean_R'][trial]
            variables['all_diff'][trial] = variables['all_mean_L'][trial] - \
                variables['all_mean_R'][trial]

        else:
            variables['V_chosen'][trial] = variables['all_mean_R'][trial]
            variables['V_unchosen'][trial] = variables['all_mean_L'][trial]
            variables['all_diff'][trial] = variables['all_mean_R'][trial] - \
                variables['all_mean_L'][trial]

        if variables['all_info_condition'][trial] == 1:
            if (variables['all_presented'][trial] < .5 and choice == 1) or (variables['all_presented'][trial] > .5 and choice == 0):
                variables['all_chosen_uncertain'][trial] = 1
            else:
                variables['all_chosen_uncertain'][trial] = 0
        else:
            variables['all_chosen_uncertain'][trial] = np.nan

        if ((variables['all_mean_R'][trial] >= variables['all_mean_L'][trial]) and choice == 1) or ((variables['all_mean_R'][trial] < variables['all_mean_L'][trial]) and choice == 0):
            variables['all_chosen_higher_mean'][trial] = 1
        else:
            variables['all_chosen_higher_mean'][trial] = 0

        confidence = data['confidence'].iloc[trial:trial+1]
        numbers_str = confidence.values[0]
        numbers_str = numbers_str.strip('[]')
        numbers = np.array([float(num.strip())
                            for num in numbers_str.split(',') if num.strip()])

        if np.any(data['immediate_judgement_orientation'] == 'vertical'):
            variables['orientation'] = 1
            # print('ok')
            variables['all_confid'][trial] = numbers[1]  # vertical
            variables['all_confid_total'][trial] = numbers[0]  # horizontal
        else:
            variables['orientation'] = 0
            variables['all_confid'][trial] = numbers[0]  # horizontal
            variables['all_confid_total'][trial] = numbers[1]  # vertical

        if 'belief_confidence_1' in data.columns:
            belief_confidence = data[f'belief_confidence_{1}'].iloc[trial:trial+1]
            numbers_str = belief_confidence.values[0]
            numbers_str = numbers_str.strip('[]')
            if numbers_str != '':
                numbers = np.array([float(num.strip())
                                    for num in numbers_str.split(',') if num.strip()])
                variables['all_belief_mean_immediate_L'][trial] = numbers[0]
                variables['all_belief_certainty_immediate_L'][trial] = numbers[1]
            else:
                variables['all_belief_mean_immediate_L'][trial] = np.nan
                variables['all_belief_certainty_immediate_L'][trial] = np.nan

            belief_confidence = data[f'belief_confidence_{2}'].iloc[trial:trial+1]
            numbers_str = belief_confidence.values[0]
            numbers_str = numbers_str.strip('[]')
            if numbers_str != '':
                numbers = np.array([float(num.strip())
                                    for num in numbers_str.split(',') if num.strip()])
                variables['all_belief_mean_immediate_R'][trial] = numbers[0]
                variables['all_belief_certainty_immediate_R'][trial] = numbers[1]
            else:
                variables['all_belief_mean_immediate_R'][trial] = np.nan
                variables['all_belief_certainty_immediate_R'][trial] = np.nan

            belief_confidence = data[f'belief_confidence_{3}'].iloc[trial:trial+1]
            numbers_str = belief_confidence.values[0]
            numbers_str = numbers_str.strip('[]')
            if numbers_str != '':
                numbers = np.array([float(num.strip())
                                    for num in numbers_str.split(',') if num.strip()])
                variables['all_belief_mean_total_L'][trial] = numbers[0]
                variables['all_belief_certainty_total_L'][trial] = numbers[1]
            else:
                variables['all_belief_mean_total_L'][trial] = np.nan
                variables['all_belief_certainty_total_L'][trial] = np.nan

            belief_confidence = data[f'belief_confidence_{4}'].iloc[trial:trial+1]
            numbers_str = belief_confidence.values[0]
            numbers_str = numbers_str.strip('[]')
            if numbers_str != '':
                numbers = np.array([float(num.strip())
                                    for num in numbers_str.split(',') if num.strip()])
                variables['all_belief_mean_total_R'][trial] = numbers[0]
                variables['all_belief_certainty_total_R'][trial] = numbers[1]
            else:
                variables['all_belief_mean_total_R'][trial] = np.nan
                variables['all_belief_certainty_total_R'][trial] = np.nan

    return variables

def get_choosing_higher_mean_rate(variables):
    V = variables

    all_diff = np.zeros_like(V['all_choice'])
    all_chosen_uncertain = np.zeros_like(V['all_choice'])
    all_chosen_higher_mean = np.zeros_like(V['all_choice'])
    for trial, choice in enumerate(V['all_choice']):
        if choice == 0:
            all_diff[trial] = V['all_mean_L'][trial] - V['all_mean_R'][trial]
        else:
            all_diff[trial] = V['all_mean_R'][trial] - V['all_mean_L'][trial]
        if V['all_info_condition'][trial] == 1:
            if (V['all_presented'][trial] < .5 and choice == 1) or (V['all_presented'][trial] > .5 and choice == 0):
                all_chosen_uncertain[trial] = 1
        if ((V['all_mean_R'][trial] > V['all_mean_L'][trial]) and choice == 1) or ((V['all_mean_R'][trial] < V['all_mean_L'][trial]) and choice == 0):
            all_chosen_higher_mean[trial] = 1
    return np.sum(all_chosen_higher_mean) / all_chosen_higher_mean.shape

def select_based_on_info(variables, info, horizon):
    V = variables
    V = V[V['all_info_condition'] == info]
    V = V[V['all_horizon'] == horizon]
    return V

def get_parameters_grid(n_params=3, N=20):

    # hard coded bounds
    X = np.linspace(-40, 40, N)
    Y = np.linspace(0.01, 40, N)
    Z = np.linspace(0, 1, N)
    H = np.linspace(0, 1, N)

    if n_params == 2:
        XX, YY = np.meshgrid(X, Y, indexing='ij')  # make a grid

        pos = np.empty(XX.shape + (2,))
        pos[:, :, 0] = XX
        pos[:, :, 1] = YY
        all_params = np.reshape(pos, [-1, 2])

    elif n_params == 3:
        XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing='ij')  # make a grid

        pos = np.empty(XX.shape + (3,))
        pos[:, :, :, 0] = XX
        pos[:, :, :, 1] = YY
        pos[:, :, :, 2] = ZZ
        all_params = np.reshape(pos, [-1, 2])

    elif n_params == 4:
        XX, YY, ZZ, HH = np.meshgrid(X, Y, Z, H, indexing='ij')  # make a grid

        pos = np.empty(XX.shape + (4,))
        pos[:, :, :, :, 0] = XX
        pos[:, :, :, :, 1] = YY
        pos[:, :, :, :, 2] = ZZ
        pos[:, :, :, :, 3] = HH
        all_params = np.reshape(pos, [-1, 4])

        # implement the upper > lower condition
        ok = np.ones(len(all_params), dtype=np.int32)
        for pp in range(len(all_params)):
            if all_params[pp, 3] <= all_params[pp, 2]:
                ok[pp] = 0
        all_params = all_params[ok.astype(bool)]
    return all_params

def fit_brute(variables, fit_choice_immed_total):
    # hand made brute function, as the original would take long with 4 params
    if fit_choice_immed_total in [0, 1, 2]:
        n_params = 3
    elif fit_choice_immed_total in [3, 4]:
        n_params = 4

    all_params = get_parameters_grid(n_params, N=20)

    # brute optimizer with constrains
    all_NLL = []
    for params in all_params:
        choice_probs = eval_logistic_model(
            variables, params, fit_choice_immed_total)
        NLL = probs_to_NLL(choice_probs, variables, fit_choice_immed_total)
        all_NLL.append(NLL)

    params = all_params[np.argmin(all_NLL)]
    NLL = all_NLL[np.argmin(all_NLL)]
    return params, NLL

def probs_to_NLL(choice_probs, variables, fit_choice_immed_total=0):
    V = variables

    like = choice_probs[np.arange(len(V['all_choice'])), V['all_choice']]
    if fit_choice_immed_total in [0, 5]:  # fitting choice
        like[like < 10**-6] = 10**-6
        NLL = - np.sum(np.log(like))
    # fitting immediate confidence
    elif np.logical_or(fit_choice_immed_total == 1, fit_choice_immed_total == 3):
        NLL = np.sqrt(np.sum((V['all_confid']/100 - like)**2))
    # fitting immediate confidence
    elif np.logical_or(fit_choice_immed_total == 2, fit_choice_immed_total == 4):
        NLL = np.sqrt(np.sum((V['all_confid_total']/100 - like)**2))
    return NLL

def get_NLL(params, variables, model_function, fit_choice_immed_total):
    # params = float(params)
    choice_probs = model_function(variables, params, fit_choice_immed_total)
    NLL = probs_to_NLL(choice_probs, variables, fit_choice_immed_total)
    return NLL

def eval_logistic_model(variables, params=[1., 1.], fit_choice_immed_total=0):
    # fit_choice_immed_total can take 4 values
    # 0 - fit choices
    # 1 - fit immediate confidence without lower and upper bounds
    # 2 - fit total confidence without lower and upper bounds
    # 3 - fit immediate confidence with lower and upper bounds
    # 4 - fit total confidence with lower and upper bounds
    
    if fit_choice_immed_total in [0, 1, 2]:
        alpha, sigma_choice, beta = params
    elif fit_choice_immed_total in [3, 4]:
        alpha, sigma_choice, lower, upper = params
    else:
        alpha, sigma_choice = params

    V = variables
    dI = np.ones_like(V['all_mean_L'])
    dI[V['all_presented'] > .5] = 1  # right shown more often, left more info
    dI[V['all_presented'] < .5] = -1  # left shown more often, right more info
    dQ = (V['all_mean_L'] - V['all_mean_R'] + alpha*dI)/sigma_choice

    p_R = expit(- dQ)
    p_R = p_R.astype(np.double)
    p_L = 1 - p_R
    choice_probs = np.concatenate(
        [p_L[:, np.newaxis], p_R[:, np.newaxis]], axis=1)

    right_informative_index = np.where(V['all_presented'] < .5)[0]
    left_informative_index = np.where(V['all_presented'] > .5)[0]

    if fit_choice_immed_total in [0, 1, 2]:
        choice_probs_distorted = choice_probs.copy()
        choice_probs_distorted[right_informative_index, 1] = (
            1-beta) * choice_probs[right_informative_index, 1]
        choice_probs_distorted[right_informative_index, 0] = choice_probs[right_informative_index,
                                                                          0] + beta * choice_probs[right_informative_index, 1]

        choice_probs_distorted[left_informative_index, 0] = (
            1-beta) * choice_probs[left_informative_index, 0]
        choice_probs_distorted[left_informative_index, 1] = choice_probs[left_informative_index,
                                                                         1] + beta * choice_probs[left_informative_index, 0]
    elif fit_choice_immed_total in [3, 4]:
        # lower and upper scaling
        # https://stackoverflow.com/questions/5294955/how-to-scale-down-a-range-of-numbers-with-a-known-min-and-max-value
        if fit_choice_immed_total > 2:
            choice_probs_distorted = lower + (upper-lower)*choice_probs
    else:
        choice_probs_distorted = choice_probs

    return choice_probs_distorted

def fit_logistic_model(variables, fit_choice_immed_total=0):

    if fit_choice_immed_total in [0, 1, 2]:
        bounds = ((-40, 40), (0.1, 40), (0., 1.))
    elif fit_choice_immed_total in [3, 4]:
        bounds = ((-40, 40), (0.1, 40), (0., 1.), (0., 1.), (0., 1.))
    else:
        bounds = ((-40, 40), (0.1, 40))
    params = brute(get_NLL, args=(
        variables, eval_logistic_model, fit_choice_immed_total), ranges=bounds, Ns=50, finish=None)
    NLL = get_NLL(params, variables, eval_logistic_model,
                  fit_choice_immed_total)
    return params, NLL

def sigmoid(x):
    return 1 / (1 + np.exp(x))