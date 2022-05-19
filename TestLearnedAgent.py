import argparse
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from DataGenerator import TSPDataset
from tqdm import tqdm
from TSPEnvironment import TSPInstanceEnv, VecEnv
from ActorCriticNetwork import ActorCriticNetwork
import time
import random


parser = argparse.ArgumentParser(description='TSPNet')

# ----------------------------------- Data ---------------------------------- #
parser.add_argument('--test_size',
                    default=512, type=int, help='Test data size')
parser.add_argument('--test_from_data',
                    default=True,
                    action='store_true', help='Render')
parser.add_argument('--n_points',
                    type=int, default=20, help='Number of points in TSP')
# ---------------------------------- Train ---------------------------------- #
parser.add_argument('--n_steps',
                    default=2000,
                    type=int, help='Number of steps in each episode')
parser.add_argument('--render',
                    action='store_true', help='Render')
# ----------------------------------- GPU ----------------------------------- #
parser.add_argument('--gpu',
                    default=True, action='store_true',
                    help='Enable gpu')
parser.add_argument('--gpu_n',
                    default=1, type=int, help='Choose GPU')
# --------------------------------- Network --------------------------------- #
parser.add_argument('--input_dim',
                    type=int, default=2, help='Input size')
parser.add_argument('--embedding_dim',
                    type=int, default=128, help='Embedding size')
parser.add_argument('--hidden_dim',
                    type=int, default=128, help='Number of hidden units')
parser.add_argument('--n_rnn_layers',
                    type=int, default=1, help='Number of LSTM layers')
parser.add_argument('--n_actions',
                    type=int, default=2, help='Number of nodes to output')
parser.add_argument('--graph_ref',
                    default=False,
                    action='store_true',
                    help='Use message passing as reference')
parser.add_argument('--random',
                    action='store_true',
                    help='random actions')
# --------------------------------- Misc --------------------------------- #
parser.add_argument('--load_path', type=str,
    default='best_policy/policy-TSP20-epoch-189.pt')
parser.add_argument('--data_dir', type=str, default='data')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        USE_CUDA = True
        print('Using GPU, %i devices available.' % torch.cuda.device_count())
        torch.cuda.set_device(args.gpu_n)
        print("GPU: %s" % torch.cuda.get_device_name(torch.cuda.current_device()))
        device = torch.device("cuda")
    else:
        USE_CUDA = False

    # loading the model from file
    if args.load_path != '' and args.random == False:
        print('  [*] Loading model from {}'.format(args.load_path))

        model = ActorCriticNetwork(args.input_dim,
                                   args.embedding_dim,
                                   args.hidden_dim,
                                   args.n_points,
                                   args.n_rnn_layers,
                                   args.n_actions,
                                   args.graph_ref)
        checkpoint = torch.load(os.path.join(os.getcwd(), args.load_path), map_location='cuda:0')
        policy = checkpoint['policy']
        model.load_state_dict(policy)
        # Move model to the GPU
        if USE_CUDA:
            model.cuda()
        model = model.eval()


    if args.test_from_data:
        # test_data = TSPDataset(dataset_fname=os.path.join(args.data_dir,
        #                                                   'TSP{}-data-test.json'
        #                                                   .format(args.n_points)),
        #                        num_samples=args.test_size, seed=1234)
        test_data = TSPDataset(dataset_fname=os.path.join(args.data_dir,
                                                          'data_train{}.json'
                                                          .format(args.n_points)),
                               size=args.n_points,
                               num_samples=args.test_size)


    test_loader = DataLoader(test_data,
                             batch_size=args.test_size,
                             shuffle=False,
                             num_workers=6)



    # run agent
    rewards = []
    best_distances = []
    step_best_distances = []
    distances = []
    initial_distances = []
    distances_per_step = []
    total_time = 0
    for batch_idx, batch_sample in enumerate(test_loader):
        b_sample = batch_sample[0].clone().detach().numpy()
        sum_reward = 0
        env = VecEnv(TSPInstanceEnv,
                     b_sample.shape[0],
                     args.n_points)
        state, initial_distance, best_state = env.reset(b_sample)
        t = 0
        hidden = None
        pbar = tqdm(total=args.n_steps)
        s_time = time.time()
        while t < args.n_steps:
            if args.render:
                env.render()
            state = torch.from_numpy(state).float()
            best_state = torch.from_numpy(best_state).float()
            if USE_CUDA:
                state = state.cuda()
                best_state = best_state.cuda()
            if args.random == False:
                with torch.no_grad():
                    _, action, _, _, _, hidden = model(state, best_state, hidden)
                    action = action.cpu().numpy()

            else:
                action = []
                for i in range(b_sample.shape[0]):
                    first = random.randint(0, args.n_points - 2)
                    second = random.randint(first, args.n_points - 1)
                    action.append([first, second])
                action = np.array(action)

            state, reward, _, best_distance, distance, best_state = env.step(action)

            sum_reward += reward
            t += 1
            step_best_distances.append(np.mean(best_distance)/10000)
            distances_per_step.append(best_distance)
            pbar.update(1)
        time_per_batch = time.time() - s_time
        total_time += time_per_batch
        pbar.close()
        rewards.append(sum_reward)
        best_distances.append(best_distance)
        distances.append(distance)
        initial_distances.append(initial_distance)

    print("Used time: {:.2f}".format(total_time))
    print("Used time per batch: {:.2f}".format(time_per_batch))
    np.save('test_recordings/2-opt-tsp20-random.pt', step_best_distances)
    avg_reward = np.mean(rewards)
    avg_best_distances = np.mean(best_distances)
    avg_initial_distances = np.mean(initial_distances)
    gap = ((avg_best_distances/10000/np.mean(test_data.opt))-1)*100

    print('Initial Cost: {:.5f} Best Cost: {:.5f} Opt Cost: {:.5f} Gap: {:.2f} %'.format(
        avg_initial_distances/10000, avg_best_distances/10000, np.mean(test_data.opt), gap))

