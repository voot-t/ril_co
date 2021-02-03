from a2c_ppo_acktr.arguments import get_args

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import pylab

## To avoid type 3 font, but this makes figure files very large. 
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

# errorfill from https://tonysyu.github.io/plotting-error-bars.html#.WRwXWXmxjZs
def errorfill(x, y, yerr, color=None, alpha_fill=0.15, ax=None, linestyle="-", linewidth = None, label=None, shade=True,dashes=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr

    if dashes is not None:
        ax.plot(x, y, linestyle=linestyle, color=color, linewidth = linewidth, label=label, dashes=dashes)
    else:
        ax.plot(x, y, linestyle=linestyle, color=color, linewidth = linewidth, label=label)
        
    if shade:
        ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def running_mean_x(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

# Except the first element (initial policy)
def running_mean(x, N):
    x_tmp = x[1:-1]
    cumsum = np.cumsum(np.insert(x_tmp, 0, 0)) 
    tmp =  (cumsum[N:] - cumsum[:-N]) / N 
    return  np.insert(tmp, 0, x[0]) 

def load(filename, limit=2084): # 20 million 
    R_test_avg = []
    step_idx = -1
    prev_step = 0
    count = 0
    try:
        with open(filename + ".txt", 'r') as f:
            for line in f:
                line = line.replace(":", " ").replace("(", " ").replace(")", " ").replace(",", " ").split()
                if step_idx == -1:
                    step_idx = line.index("Step") + 1
                    R_test_avg_idx = line.index("[R_te]") + 6
                cur_step = int(line[step_idx]) 
                if cur_step < prev_step:
                    print("reset array at %s" % filename)
                    R_test_avg = [] 
                R_test_avg += [ float(line[R_test_avg_idx]) ]
                prev_step = cur_step 
                count += 1
                if limit != 0 and count >= limit:
                    break 
        R_test_avg = np.reshape(np.array(R_test_avg), (-1, 1))   # [iter , 1] array
        return R_test_avg
    except  Exception as e:
        print(e)
        return np.reshape(np.array([-999]), (-1, 1))

def plot():
    args = get_args()

    env_name = args.env_name 
    """ get config about trajectory file"""

    from a2c_ppo_acktr.algo import ail 
    discr = ail.AIL(0,0,0, args, True)
    m_return_list = discr.m_return_list

    # main plot with policy snapshots in main paper
    plot_methods = ["ril_co_apl", "ail_apl", "ail_logistic", "ail_unhinged", "fairl", "vild", "bc"] 
    plot_aug = ""

    # ## main plot with gaussian noise in main paper
    # plot_methods = ["ril_co_apl", "ail_apl", "ail_logistic", "ail_unhinged", "fairl", "vild"] 
    # plot_aug = ""

    ## ablation in appendix 
    # plot_methods = ["ril_co_apl", "ril_co_logistic", "ril_apl", "ril_logistic"]  
    # plot_aug = "_app"

    seed_list = [1, 2, 3, 4, 5]
    # seed_list = [1,1]
    seed_num = len(seed_list)
    print(seed_list)

    if args.plot_each:
        plot_np = [args.noise_prior]
    else:
        plot_np = [0.0, 0.1, 0.2, 0.3, 0.4]

    R_np_mean = np.zeros((len(plot_methods), len(plot_np)))
    R_np_std =  np.zeros((len(plot_methods), len(plot_np)))

    limit = 2048
    # limit = 0

    max_len = 10000
    plot_large = args.plot_large

    for np_i in range(0, len(plot_np)):

        R_test_all = []
        gail_legend = []
        c_tmp = []
        l_tmp = []

        if args.noise_type == "policy" or plot_np[np_i]  == 0.0:
            traj_name = "np%0.1f" % plot_np[np_i] 
        elif args.noise_type == "action":
            traj_name = "na%0.1f" % plot_np[np_i] 
        if args.traj_deterministic:
            traj_name += "_det"
        else:
            traj_name += "_sto" 
            
        if any("ril_co_apl" in s for s in plot_methods):
            """ RIL_CO """
            R_test_avg = []
            cat = 1
            args.il_algo = "ril_co"
            args.ail_loss_type = "apl"
            args.ail_saturate = 1
            method_name = args.algo.upper() + "_" + args.il_algo.upper() 
            hypers = "ec%0.5f" % args.entropy_coef + "_gp%0.3f" % args.gp_lambda
            if "RIL" in args.il_algo.upper() :
                hypers += "_%s_sat%d" % (args.ail_loss_type, args.ail_saturate)
                # hypers += "_%s_sat%d_ver0" % (args.ail_loss_type, args.ail_saturate)
            if args.reward_std: hypers += "_rs"

            for seed in seed_list:
                exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

                print(filename) 

                R_test_avg_i = load(filename, limit)
                if R_test_avg_i[0, 0] == -999:
                    cat = 0
                    print("cannot load %s" % exp_name) 
                else:
                    load_iter = R_test_avg_i.shape[0]
                    if load_iter < max_len:
                        max_len = load_iter 
                        for i in range(0, seed-1):
                            R_test_avg[i] = R_test_avg[i][0:max_len, :]
                    R_test_avg_i = R_test_avg_i[0:max_len, :]
                    R_test_avg += [R_test_avg_i]
            if cat:
                R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
                R_test_all += [R_test_avg]
                gail_legend += ["RIL-Co (AP)"]
                c_tmp += ["r"]
                l_tmp += ["-"]

        if any("ril_co_logistic" in s for s in plot_methods):
            """ RIL_CO """
            R_test_avg = []
            cat = 1
            args.il_algo = "ril_co"
            args.ail_loss_type = "logistic"
            args.ail_saturate = 1
            method_name = args.algo.upper() + "_" + args.il_algo.upper() 
            hypers = "ec%0.5f" % args.entropy_coef + "_gp%0.3f" % args.gp_lambda
            if "RIL" in args.il_algo.upper() :
                hypers += "_%s_sat%d" % (args.ail_loss_type, args.ail_saturate)
                # hypers += "_%s_sat%d_ver0" % (args.ail_loss_type, args.ail_saturate)
            if args.reward_std: hypers += "_rs"

            for seed in seed_list:
                exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

                print(filename) 

                R_test_avg_i = load(filename, limit)
                if R_test_avg_i[0, 0] == -999:
                    cat = 0
                    print("cannot load %s" % exp_name) 
                else:
                    load_iter = R_test_avg_i.shape[0]
                    if load_iter < max_len:
                        max_len = load_iter 
                        for i in range(0, seed-1):
                            R_test_avg[i] = R_test_avg[i][0:max_len, :]
                    R_test_avg_i = R_test_avg_i[0:max_len, :]
                    R_test_avg += [R_test_avg_i]
            if cat:
                R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
                R_test_all += [R_test_avg]
                gail_legend += ["RIL-Co (logistic)"]
                c_tmp += ["blue"]
                l_tmp += [":"]

        if any("ril_apl" in s for s in plot_methods):
            """ RIL """
            R_test_avg = []
            cat = 1
            args.il_algo = "ril"
            args.ail_loss_type = "apl"
            args.ail_saturate = 1
            method_name = args.algo.upper() + "_" + args.il_algo.upper() 
            hypers = "ec%0.5f" % args.entropy_coef + "_gp%0.3f" % args.gp_lambda
            if "RIL" in args.il_algo.upper() :
                hypers += "_%s_sat%d" % (args.ail_loss_type, args.ail_saturate)
                # hypers += "_%s_sat%d_ver0" % (args.ail_loss_type, args.ail_saturate)
            if args.reward_std: hypers += "_rs"

            for seed in seed_list:
                exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

                print(filename) 

                R_test_avg_i = load(filename, limit)
                if R_test_avg_i[0, 0] == -999:
                    cat = 0
                    print("cannot load %s" % exp_name) 
                else:
                    load_iter = R_test_avg_i.shape[0]
                    if load_iter < max_len:
                        max_len = load_iter 
                        for i in range(0, seed-1):
                            R_test_avg[i] = R_test_avg[i][0:max_len, :]
                    R_test_avg_i = R_test_avg_i[0:max_len, :]
                    R_test_avg += [R_test_avg_i]
            if cat:
                R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
                R_test_all += [R_test_avg]
                gail_legend += ["RIL-P (AP)"]
                c_tmp += ["m"]
                l_tmp += ["-."]

        if any("ril_logistic" in s for s in plot_methods):
            """ RIL """
            R_test_avg = []
            cat = 1
            args.il_algo = "ril"
            args.ail_loss_type = "logistic"
            args.ail_saturate = 1
            method_name = args.algo.upper() + "_" + args.il_algo.upper() 
            hypers = "ec%0.5f" % args.entropy_coef + "_gp%0.3f" % args.gp_lambda
            if "RIL" in args.il_algo.upper() :
                hypers += "_%s_sat%d" % (args.ail_loss_type, args.ail_saturate)
                # hypers += "_%s_sat%d_ver0" % (args.ail_loss_type, args.ail_saturate)
            if args.reward_std: hypers += "_rs"

            for seed in seed_list:
                exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

                print(filename) 

                R_test_avg_i = load(filename, limit)
                if R_test_avg_i[0, 0] == -999:
                    cat = 0
                    print("cannot load %s" % exp_name) 
                else:
                    load_iter = R_test_avg_i.shape[0]
                    if load_iter < max_len:
                        max_len = load_iter 
                        for i in range(0, seed-1):
                            R_test_avg[i] = R_test_avg[i][0:max_len, :]
                    R_test_avg_i = R_test_avg_i[0:max_len, :]
                    R_test_avg += [R_test_avg_i]
            if cat:
                R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
                R_test_all += [R_test_avg]
                gail_legend += ["RIL-P (logistic)"]
                c_tmp += ["darkgreen"]
                l_tmp += ["--"]

        if any("ail_logistic" in s for s in plot_methods):
            """ AIL logistic (GAIL) """
            R_test_avg = []
            cat = 1
            args.il_algo = "ail"
            args.ail_loss_type = "logistic"
            args.ail_saturate = 1
            method_name = args.algo.upper() + "_" + args.il_algo.upper() 
            hypers = "ec%0.5f" % args.entropy_coef + "_gp%0.3f" % args.gp_lambda
            if "AIL" in args.il_algo.upper():
                hypers += "_%s_sat%d" % (args.ail_loss_type, args.ail_saturate)  
            if args.reward_std: hypers += "_rs"      

            for seed in seed_list:
                exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

                print(filename) 

                R_test_avg_i = load(filename, limit)
                if R_test_avg_i[0, 0] == -999:
                    cat = 0
                    print("cannot load %s" % exp_name) 
                else:
                    load_iter = R_test_avg_i.shape[0]
                    if load_iter < max_len:
                        max_len = load_iter 
                        for i in range(0, seed-1):
                            R_test_avg[i] = R_test_avg[i][0:max_len, :]
                    R_test_avg_i = R_test_avg_i[0:max_len, :]
                    R_test_avg += [R_test_avg_i]
            if cat:
                R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
                R_test_all += [R_test_avg]
                gail_legend += ["GAIL (logistic)"]
                c_tmp += ["goldenrod"]
                l_tmp += ["--"]

        if any("ail_unhinged" in s for s in plot_methods):
            """ AIL unhinged (W-GAIL) """
            R_test_avg = []
            cat = 1
            args.il_algo = "ail"
            args.ail_loss_type = "unhinged"
            args.ail_saturate = 0
            method_name = args.algo.upper() + "_" + args.il_algo.upper() 
            hypers = "ec%0.5f" % args.entropy_coef + "_gp%0.3f" % args.gp_lambda
            if "AIL" in args.il_algo.upper():
                hypers += "_%s_sat%d" % (args.ail_loss_type, args.ail_saturate)        
            if args.reward_std: hypers += "_rs"

            for seed in seed_list:
                exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

                # print(filename) 

                R_test_avg_i = load(filename, limit)
                if R_test_avg_i[0, 0] == -999:
                    cat = 0
                    print("cannot load %s" % exp_name) 
                else:
                    load_iter = R_test_avg_i.shape[0]
                    if load_iter < max_len:
                        max_len = load_iter 
                        for i in range(0, seed-1):
                            R_test_avg[i] = R_test_avg[i][0:max_len, :]
                    R_test_avg_i = R_test_avg_i[0:max_len, :]
                    R_test_avg += [R_test_avg_i]
            if cat:
                R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
                R_test_all += [R_test_avg]
                gail_legend += ["GAIL (unhinged)"]
                c_tmp += ["darkgreen"]
                l_tmp += [":"]

        if any("ail_apl" in s for s in plot_methods):
            """ AIL apl """
            R_test_avg = []
            cat = 1
            args.il_algo = "ail"
            args.ail_loss_type = "apl"
            args.ail_saturate = 1
            method_name = args.algo.upper() + "_" + args.il_algo.upper() 
            hypers = "ec%0.5f" % args.entropy_coef + "_gp%0.3f" % args.gp_lambda
            if "AIL" in args.il_algo.upper():
                hypers += "_%s_sat%d" % (args.ail_loss_type, args.ail_saturate)       
            if args.reward_std: hypers += "_rs"

            for seed in seed_list:
                exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

                print(filename) 
                
                R_test_avg_i = load(filename, limit)
                if R_test_avg_i[0, 0] == -999:
                    cat = 0
                    print("cannot load %s" % exp_name) 
                else:
                    load_iter = R_test_avg_i.shape[0]
                    if load_iter < max_len:
                        max_len = load_iter 
                        for i in range(0, seed-1):
                            R_test_avg[i] = R_test_avg[i][0:max_len, :]
                    R_test_avg_i = R_test_avg_i[0:max_len, :]
                    R_test_avg += [R_test_avg_i]
            if cat:
                R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
                R_test_all += [R_test_avg]
                gail_legend += ["GAIL (AP)"]
                c_tmp += ["blue"]
                l_tmp += ["-"]

        if any("fairl" in s for s in plot_methods):
            """ FAIRL """
            R_test_avg = []
            cat = 1
            args.il_algo = "fairl"
            method_name = args.algo.upper() + "_" + args.il_algo.upper() 
            hypers = "ec%0.5f" % args.entropy_coef + "_gp%0.3f" % args.gp_lambda
            if args.reward_std: hypers += "_rs"

            for seed in seed_list:
                exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

                print(filename) 

                R_test_avg_i = load(filename, limit)
                if R_test_avg_i[0, 0] == -999:
                    cat = 0
                    print("cannot load %s" % exp_name) 
                else:
                    load_iter = R_test_avg_i.shape[0]
                    if load_iter < max_len:
                        max_len = load_iter 
                        for i in range(0, seed-1):
                            R_test_avg[i] = R_test_avg[i][0:max_len, :]
                    R_test_avg_i = R_test_avg_i[0:max_len, :]
                    R_test_avg += [R_test_avg_i]
            if cat:
                R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
                R_test_all += [R_test_avg]
                gail_legend += ["FAIRL"]
                c_tmp += ["violet"]
                l_tmp += ["--"]

        if any("vild" in s for s in plot_methods):
            """ VILD """
            R_test_avg = []
            cat = 1
            args.il_algo = "vild"
            method_name = args.algo.upper() + "_" + args.il_algo.upper() 
            hypers = "ec%0.5f" % args.entropy_coef + "_gp%0.3f" % args.gp_lambda
            hypers += "_%s" % ("logistic")
            if args.reward_std: hypers += "_rs"

            for seed in seed_list:
                exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

                print(filename) 

                R_test_avg_i = load(filename, limit)
                if R_test_avg_i[0, 0] == -999:
                    cat = 0
                    print("cannot load %s" % exp_name) 
                else:
                    load_iter = R_test_avg_i.shape[0]
                    if load_iter < max_len:
                        max_len = load_iter 
                        for i in range(0, seed-1):
                            R_test_avg[i] = R_test_avg[i][0:max_len, :]
                    R_test_avg_i = R_test_avg_i[0:max_len, :]
                    R_test_avg += [R_test_avg_i]
            if cat:
                R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
                R_test_all += [R_test_avg]
                gail_legend += ["VILD"]
                c_tmp += ["deepskyblue"]
                l_tmp += ["--"]

        if any("bc" in s for s in plot_methods):
            """ BC """
            R_test_avg = []
            cat = 1
            args.il_algo = "bc"
            method_name = args.il_algo.upper() 
            hypers = "bc"

            for seed in seed_list:
                exp_name = "%s-%s-%s_s%d" % (traj_name, method_name, hypers, seed)
                filename = "./results_IL/%s/%s/%s-%s" % (method_name, env_name, env_name, exp_name)

                print(filename) 

                R_test_avg_i = load(filename, limit)
                if R_test_avg_i[0, 0] == -999:
                    cat = 0
                    print("cannot load %s" % exp_name) 
                else:
                    load_iter = R_test_avg_i.shape[0]
                    if load_iter < max_len:
                        max_len = load_iter 
                        for i in range(0, seed-1):
                            R_test_avg[i] = R_test_avg[i][0:max_len, :]
                    R_test_avg_i = R_test_avg_i[0:max_len, :]
                    R_test_avg += [R_test_avg_i]
            if cat:
                R_test_avg = np.concatenate(R_test_avg, axis=1) # array [iter, test_episode * seed]
                R_test_all += [R_test_avg]
                gail_legend += ["BC"]
                c_tmp += ["k"]
                l_tmp += ["-"]


        """ Compute statistics for plotting """
        R_test_mean = []
        R_test_std = []
        R_test_last = []
        R_test_last_mean = []
        R_test_last_std = []
        best_idx = -1 
        best_R = -1e6
        for i in range(0, len(plot_methods)):
            R_test = R_test_all[i]
            # print(gail_legend[i])
            print(R_test.shape) 

            """ This is for plotting """
            R_test_mean += [np.mean(R_test, 1)]
            R_test_std += [np.std(R_test, 1)/np.sqrt(R_test.shape[1])]
        
            ## Average last final X iteration. 
            X = 50  # last 500000 steps. This is approximately 1000 iterations since ACKTR uses 640 steps per training iteration.
            R_test_last += [np.mean(R_test[-X:,:], 0)]

            last_mean = np.mean(R_test_last[i], 0)
            if last_mean > best_R:
                best_idx = i 
                best_R = last_mean 

            R_test_last_mean += [last_mean]
            R_test_last_std += [np.std(R_test_last[i], 0)/np.sqrt(R_test.shape[1])]

            R_np_mean[i, np_i] = last_mean
            R_np_std[i, np_i] = np.std(R_test_last[i], 0)/np.sqrt(R_test.shape[1])

            ### Print statistics 
            # print("%s: %0.1f(%0.1f)" % (gail_legend[i], R_test_mean_last, R_test_std_last))


        """ For t_test """
        ## paired t-test
        from scipy import stats
        
        best_m = R_test_last[best_idx]
        p_list = []
        for i in range(0, len(plot_methods)):
            # if gail_legend[i] == "InfoGAIL (best)": continue 
            if i != best_idx:
                _, p = stats.ttest_ind(best_m, R_test_last[i], nan_policy="omit")
            else:
                p = 1        
            p_list += [p]
            
        ## copied able latex format
        latex = ""
        for i in range(0, len(plot_methods)):
            # if gail_legend[i] == "InfoGAIL (best)": continue 
            print("%-70s:   Sum %0.0f(%0.0f) with p-value %f" % (gail_legend[i], R_test_last_mean[i], R_test_last_std[i], p_list[i]))
            if p_list[i] > 0.01:    # 1 percent confidence 
                latex += " & \\textbf{%0.0f (%0.0f)}" % (R_test_last_mean[i], R_test_last_std[i])
            else:
                latex += " & %0.0f (%0.0f)" % (R_test_last_mean[i], R_test_last_std[i])
        print(latex + " \\\\")

        if args.plot_each:
            skipper = 50  # Use sliding window to compute running mean and std for clearer figures. skipper 1 = no running.
            running = 1
            x_max_iter = len(R_test_mean[0])

            """ Plot """
            if plot_large: 
                linewidth = 1+plot_large
                fontsize = 21
                f = plt.figure(figsize=(8, 6)) 
            else:
                linewidth = 1 
                fontsize = 14   
                f = plt.figure(figsize=(4, 3)) 

            ax = f.gca()

            """ Plot """
            cc_tmp =  ["k"] + ["black"] + ["gray"] + ["darkgray"] + ["dimgray"] + ["silver"] 
            
            for i in range(0, len(R_test_all)):
                if running:
                    y_plot = running_mean(R_test_mean[i][:x_max_iter], skipper)
                    y_err = running_mean(R_test_std[i][:x_max_iter], skipper)
                else:
                    y_plot = R_test_mean[i][:x_max_iter:skipper]
                    y_err = R_test_std[i][:x_max_iter:skipper]

                x_ax = np.arange(0, len(y_plot)) * 10000 
                
                if gail_legend[i] == "FAIRL":
                    errorfill(x_ax, y_plot, yerr=y_err, color=c_tmp[i], linestyle=l_tmp[i], linewidth=linewidth, label=gail_legend[i], shade=1, dashes=(3,1,1,1))
                elif gail_legend[i] == "VILD":
                    errorfill(x_ax, y_plot, yerr=y_err, color=c_tmp[i], linestyle=l_tmp[i], linewidth=linewidth, label=gail_legend[i], shade=1, dashes=(4,1))
                else:
                    errorfill(x_ax, y_plot, yerr=y_err, color=c_tmp[i], linestyle=l_tmp[i], linewidth=linewidth, label=gail_legend[i], shade=1)

            if plot_large == 1 :
            #     ax.legend(prop={"size":fontsize-4}, frameon=True, framealpha=1, ncol=2, loc='lower right')  
                ax.legend(prop={"size":fontsize-4}, frameon=True, framealpha=1, ncol=2, loc="upper left", bbox_to_anchor=(0.19,0.52))
                # ax.legend(prop={"size":fontsize}, frameon=True, framealpha=1, ncol=1, loc='lower right')  # slide
                # ax.legend(prop={"size":fontsize}, frameon=True, framealpha=1, ncol=1, loc='upper left')  # slide

            if args.noise_type == "policy":
                plot_name = "%s_%s_np%0.1f" % (args.algo.upper(), env_name, plot_np[np_i])
            elif args.noise_type == "action":
                plot_name = "%s_%s_na%0.1f" % (args.algo.upper(), env_name, plot_np[np_i])
                
            title = "%s" % (env_name[:-12])

            if plot_large:
                plt.title(title, fontsize=fontsize+1)
            else:
                plt.title(title, fontsize=fontsize)
                    
            plt.xlabel("Transition samples", fontsize=fontsize-2)      
            plt.ylabel('Cumulative rewards', fontsize=fontsize-2)
            # plt.xticks(fontsize=fontsize-2)
            plt.xticks([0, 5e6, 10e6, 15e6, 20e6], fontsize=fontsize-2)

            plt.yticks(fontsize=fontsize-2)
            plt.tight_layout()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

            ## plot legend.
            if plot_large == 2:
                figlegend = pylab.figure(figsize=(22,1))
                #pylab.figlegend(*ax.get_legend_handles_labels(), prop={"size":fontsize-3}, ncol=7, handlelength=2.8, )
                pylab.figlegend(*ax.get_legend_handles_labels(), prop={"size":fontsize-3}, ncol=8, handlelength=2.0, )       
                figlegend.savefig("./figures/legend%s.pdf" % plot_aug, bbox_inches='tight', )


    """ Plot statistics for all np """
    if not args.plot_each:
        """ Plot """
        if plot_large: 
            linewidth = 3
            fontsize = 21
            f = plt.figure(figsize=(8, 6)) 
        else:
            linewidth = 2 
            fontsize = 14   
            f = plt.figure(figsize=(4, 3))

        ax = f.gca()

        """ Plot """
        cc_tmp =  ["k"] + ["black"] + ["gray"] + ["darkgray"] + ["dimgray"] + ["silver"] 
        
        for i in range(0, len(plot_methods)):
            y_mean_all = R_np_mean[i,:]
            y_std_all = R_np_std[i,:]
            x_ax = np.arange(0, len(y_mean_all)) * 0.1

            # if i == 0 and m_return_list is not None :
            #     # for i_k in range(0, len(m_return_list)):
            #     for i_k in [0]:
            #         opt_plot = (x_ax * 0) + m_return_list[i_k]
            #         ax.plot(x_ax, opt_plot, color=cc_tmp[i_k], linestyle=":", linewidth=linewidth+0.5)
                    
            if gail_legend[i] == "FAIRL":
                errorfill(x_ax, y_mean_all, yerr=y_std_all, color=c_tmp[i], linestyle=l_tmp[i], linewidth=linewidth, label=gail_legend[i], shade=1, dashes=(3,1,1,1))
            elif gail_legend[i] == "VILD":
                errorfill(x_ax, y_mean_all, yerr=y_std_all, color=c_tmp[i], linestyle=l_tmp[i], linewidth=linewidth, label=gail_legend[i], shade=1, dashes=(3,3))
            else:
                errorfill(x_ax, y_mean_all, yerr=y_std_all, color=c_tmp[i], linestyle=l_tmp[i], linewidth=linewidth, label=gail_legend[i], shade=1)
                

        plot_name = "%s_%s" % (args.algo.upper(), env_name)

        # title = "%s" % (env_name[:-3])
        title = "%s" % (env_name[:-12]) # Remove "BulletEnv-v0" from plot title

        if plot_large:
            ax.legend(prop={"size":fontsize-3}, frameon=True, loc = 'lower left', framealpha=1, ncol=1)   
            plt.title(title, fontsize=fontsize+1)
        else:
            plt.title(title, fontsize=fontsize)
                    
        plt.xlabel("Noise rate", fontsize=fontsize-2)      
        plt.ylabel('Cumulative rewards', fontsize=fontsize-2)
        plt.xticks(fontsize=fontsize-2)
        plt.yticks(fontsize=fontsize-2)
        plt.tight_layout()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    fig_name = "./figures/%s" % (plot_name)
    if plot_large: 
        fig_name += "_large"
    fig_name += plot_aug
    print(fig_name) 

    if args.plot_save:
        f.savefig(fig_name + ".pdf", bbox_inches='tight', pad_inches = 0)
    
    if args.plot_show:
        plt.show()

if __name__ == "__main__":
    plot()
