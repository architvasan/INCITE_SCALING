import matplotlib.pyplot as plt
import pandas as pd
ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
folds_p_hr_list = []
ns_p_day_list = []
for r in ranks:
    try:
        df = pd.read_csv(f'Folding/folding_times_{r}.csv')
    except:
        continue

    duration_prnk = df['duration_seconds'].max()
    num_folds = 2*len(df)
    #print(num_folds)
    folds_p_hr = 60 * 60 * (num_folds/duration_prnk)
    #print(folds_p_hr)
    folds_p_hr_list.append(folds_p_hr)
for r in ranks:
    try:
        df = pd.read_csv(f'Simulation/simulate_implicit/simulation_times_{r}.csv')
    except:
        continue

    duration_prnk = df['duration_seconds'].max()
    num_ns = 0.2*len(df)
    print(num_ns)
    folds_p_hr = 24 * 60 * 60 * (num_ns/duration_prnk)
    print(folds_p_hr)
    ns_p_day_list.append(folds_p_hr)

plt.scatter(ranks, folds_p_hr_list)
plt.plot(ranks, folds_p_hr_list)
plt.savefig('folds_p_hr_scaling.png', bbox_inches='tight')
plt.close()

plt.scatter(ranks, ns_p_day_list)
plt.plot(ranks, ns_p_day_list)
plt.savefig('ns_p_day_scaling.png', bbox_inches='tight')
plt.close()
