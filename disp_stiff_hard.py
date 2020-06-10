import os
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import matplotlib.pyplot as plt
from models import DISP_STIFF_UQPINN
from data import DatasetLoader
import tensorflow as tf
from argparse import ArgumentParser

np.random.seed(1234)
tf.random.set_seed(1234)


def plot_displacement_mean_std(X, real_samples, generated_samples,
                               y_label, title,save_name):
    '''
    Same function as the 1D normal case
    '''

    real_samples = np.squeeze(real_samples)
    generated_samples = np.squeeze(generated_samples)
    plt.rcParams.update({'font.size': 16})
    plt.xlim(0, np.max(X))
    plt.ylim(min(np.min(real_samples), np.min(generated_samples)),
             max(np.max(real_samples), np.max(generated_samples)))
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel(y_label)

    std_mult = 2
    # Calculate mean and standard deviation
    f_mean = tf.reduce_mean(real_samples, axis=0)
    f_std = tf.math.reduce_std(real_samples, axis=0)
    avg_f_std = tf.reduce_mean(f_std)

    g_mean = tf.reduce_mean(generated_samples, axis=0)
    g_std = tf.math.reduce_std(generated_samples, axis=0)
    avg_g_std = tf.reduce_mean(g_std)

    avg_loss = tf.keras.losses.mse(f_mean, g_mean)
    std_loss = tf.keras.losses.mse(f_std, g_std)
    print(f"Real {y_label} Average std: {avg_f_std}")
    print(f"Generated {y_label} Average std: {avg_g_std}")
    print(f"Mean {y_label} L2 Error: {avg_loss}")
    print(f"STD  {y_label} L2 Error: {std_loss}")
    # Plot mean and standard deviation
    plt.fill_between(tf.squeeze(X), f_mean - (std_mult * f_std),
                     f_mean + (std_mult * f_std), alpha=0.3, color='b',
                     label='Target Std')
    plt.fill_between(tf.squeeze(X), g_mean - (std_mult * g_std),
                     g_mean + (std_mult * g_std), alpha=0.3, color='r',
                     label='Generated Std')
    plt.plot(X, f_mean, 'b', label='Target Mean')
    plt.plot(X, g_mean, 'r', label='Generated Mean')
    save_name = str(save_name) + f"/{y_label}_displacement.png"
    print(save_name)
    plt.savefig(save_name)
def plot_z_hist(Z,indicies,save_name):
    '''
    Takes list of indicies to take an slice of the Z samples, and plot their
    histogram
    Should look like an normal distribution
    '''
    num_cols=2
    fig = plt.figure(figsize=(40,40))
    for ct,idx in enumerate(indicies):
        fig.add_subplot(len(indicies)/num_cols,num_cols,ct+1)
        plt.hist(Z[idx,:],bins=50)
    save_name = str(save_name) + "/z_histograms.png"
    print(save_name)
    plt.savefig(save_name)
if __name__ == "__main__":
    parser = ArgumentParser(description="Adds some helpful control")
    parser.add_argument('-ts', type=int, default=20000, help="training steps")
    parser.add_argument('-debug', type=bool,
                        help="toggles training steps and pdb")
    parser.add_argument('-run_name', type=str, default="",
                        help="optional name for the run")
    parser.add_argument('-num_ckpts',default=4,type=int,help='num of save ckpts')
    parser.add_argument('-lam', type=float,default=1.5)
    parser.add_argument('-beta_1',type=float,default=.9)
    parser.add_argument('-beta',type=float,default=1.0)
    args = parser.parse_args()
    run_name = Path.home()/"pigan_examples/UQPINN/runs" / args.run_name
    TRAINING_STEPS = args.ts
    SAVE_FREQ = int(TRAINING_STEPS/args.num_ckpts)
    if args.debug:
        tf.config.experimental_run_functions_eagerly(True)
        TRAINING_STEPS = 10
        SAVE_FREQ = 5
        run_name = Path.home()/"pigan_examples/UQPINN/runs/debug4/"

    run_name.mkdir(parents=True, exist_ok=True)
    print(run_name)
    #Number of collocation points
    N_f = 100

    # Number of testing points
    N_ff = 200

    # Number of the training data
    N_u = 60
    # Number of snapshots
    N_s = 100
    # Number of testing samples
    N_ts = 1000

    # Load train and test dataset
    dataset_loader = DatasetLoader(1.0, 1.5)
    X_u, X_f, Y_u = dataset_loader.generate_train_data(N_u, N_f, N_s)
    X_t, u_test, e_test = dataset_loader.generate_test_data()

    #Create model
    model = DISP_STIFF_UQPINN(run_name,args.beta_1, lam=args.lam,
                              beta=args.beta)
    #Train Model
    model.train(TRAINING_STEPS, N_u, N_f, N_s, X_f, X_u, Y_u,X_t,SAVE_FREQ)
    #Save Model
    model.save(run_name)
    #Generate samples for visualization
    gen_u, gen_e,gen_z = model.generate_samples(X_t, N_ts)
    #Plot Z to check how the encoder does
    plot_z_hist(gen_z,[10,20,30,40],run_name)
    plt.figure()
    #Plot U
    plot_displacement_mean_std(X_t, u_test, gen_u, "U", "Displacement", run_name)
    plt.figure()
    #Plot E
    plot_displacement_mean_std(X_t, e_test, gen_e, "E", "Displacement", run_name)

