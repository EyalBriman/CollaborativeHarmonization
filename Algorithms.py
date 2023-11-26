import math
import matplotlib.pyplot as plt
import random
import numpy as np
from time import time
from two_gram import *
from DistanceOfTwoChords import chords_distances

max_iters = 10000
max_time = 60


def change_chords(W):
    W = W.copy()
    place = np.random.randint(0, len(W))
    W[place] = (W[place] + np.random.randint(1, len(chords_distances))) % len(chords_distances)
    return W


def change_chords_clustering(partition, W, agent_clustering, upper_bound):
    W = W.copy()
    partition = partition.copy()
    agent_clustering = agent_clustering.copy()
    flag = random.choices(range(4), weights=[0.4, 0.3, 0.2, 0.1])[0]
    if flag == 0:
        place = np.random.randint(0, len(W))
        W[place] = (W[place] + np.random.randint(1, len(chords_distances))) % len(chords_distances)
    elif flag == 1:
        place = np.random.randint(0, len(agent_clustering))
        agent_clustering[place] = random.choice(partition)
    elif len(partition) > 1 and flag == 2:
        place = np.random.randint(1, len(partition))
        if place < len(partition) - 1:
            value = random.randint(partition[place - 1] + 1, partition[place + 1] - 1)
        else:
            value = random.randint(partition[place - 1] + 1, len(W) - 1)
        partition[place] = value
    else:
        new_partition_size = random.randint(0, upper_bound)
        partition = [0] + random.sample(range(1, len(W)), new_partition_size)
        partition.sort()
    return partition, W, agent_clustering


def song_distance(song1, song2):
    return sum([chords_distances[song1[i]][song2[i]] for i in range(len(song1))]) / len(song1)


def get_chords_probability(chords):
    probabilities = []
    for i in range(len(chords) - 1):
        try:
            probabilities.append(probs[chords[i]][chords[i + 1]])
        except KeyError:
            probabilities.append(lepsilon)
    return combine_probabilities(probabilities)


def get_ints(x):
    return [int(y) for y in x]


def get_2gram_score(song):
    return get_chords_probability(song)


def proportional_target_func(songs, W):
    res = 0
    for i in range(len(songs)):
        dist = np.sort([chords_distances[W[j]][songs[i][j]] for j in range(len(W))])
        res += sum([dist[j] / (j + 1) for j in range(len(dist))])
    return res / len(songs) / len(W)


def proportional_2gram_target_func(songs, W):
    score1 = proportional_target_func(songs, W)
    score2 = get_2gram_score(W)
    return score1 - 2e-4 * score2


def majority_algorithm(songs):
    song_len = len(songs[0])
    majority = numpy.zeros((len(chords_distances), song_len))

    for song in songs:
        for i, chord in enumerate(song):
            majority[chord, i] += 1
    return list(numpy.argmax(majority, axis=0))


def proportional_algorithm(songs, iters=max_iters, init_with_majority=True):
    if init_with_majority:
        init = majority_algorithm(songs)
    else:
        init = numpy.random.randint(0, len(chords_distances), len(songs[0]))

    return simulated_annealing(songs, init, iters, iters / 4, proportional_target_func)


def kemeny(songs):
    W = []
    for i in range(len(songs[0])):
        max_d = len(songs)
        best_chord = -1
        for j in range(len(chords_distances)):
            d = sum([chords_distances[songs[k][i]][j] for k in range(len(songs))])
            if d < max_d:
                max_d = d
                best_chord = j
        W.append(best_chord)
    return W


def proportional_2gram_algorithm(songs, iters=max_iters, init_with_majority=True):
    if init_with_majority:
        init = majority_algorithm(songs)
    else:
        init = numpy.random.randint(0, len(chords_distances), len(songs[0]))

    return simulated_annealing(songs, init, iters, iters / 4, proportional_2gram_target_func)


def kemeny_2gram(songs):
    global probs
    W = []
    for i in range(len(songs[0])):
        base_T = []
        for j in range(len(chords_distances)):
            base_T.append(sum([chords_distances[songs[k][i]][j] for k in range(len(songs))]))
        if i > 0:
            new_T = []
            min_d = math.inf
            for j in range(len(chords_distances)):
                for m in range(len(chords_distances)):
                    d = base_T[j] + T[m] - 0.1 * probs[m][j]
                    if d < min_d:
                        min_d = d
                new_T.append(min_d)
        else:
            new_T = base_T
        T = new_T.copy()
        W.append(T.index(min(T)))
    return W


def majority_2gram(songs):
    global probs
    W = []
    for i in range(len(songs[0])):
        base_T = [0] * len(chords_distances)
        for j in range(len(chords_distances)):
            for song in songs:
                base_T[song[i]] -= 1 / len(songs)
        if i > 0:
            new_T = []
            min_d = math.inf
            for j in range(len(chords_distances)):
                for m in range(len(chords_distances)):
                    d = base_T[j] + T[m] - probs[m][j]
                    if d < min_d:
                        min_d = d
                new_T.append(min_d)
        else:
            new_T = base_T
        T = new_T.copy()
        W.append(T.index(min(T)))
    return W


def P(j, z, partition, k):
    if z < len(partition) - 1:
        return 1 if partition[z] <= j < partition[z + 1] else 0
    else:
        return 1 if partition[z] <= j <= k else 0


def Q(i, z, agent_clustering):
    if agent_clustering[i] == z:
        return 1
    else:
        return 0.5


def kemeny_clustering_target_func(songs, partition, W, agent_clustering):
    total_distance = 0
    total_pq = 0
    for z in range(len(partition)):
        for i in range(len(songs)):
            for j in range(len(W)):
                pq = P(j, z, partition, len(W)) * Q(i, partition[z], agent_clustering)
                total_pq += pq
                total_distance += pq * chords_distances[songs[i][j]][W[j]]
    return total_distance / total_pq


def kemeny_clustering_2gram_target_func(songs, partition, W, agent_clustering):
    score1 = kemeny_clustering_target_func(songs, partition, W, agent_clustering)
    score2 = get_2gram_score(W)
    return score1 - 0.1 * score2


def simulated_annealing_clustering(songs, partition, W, agent_clustering, iters, upper_bound, success_limit,
                                   target_func):
    current_partition = partition
    current_W = W
    current_agent_clustering = agent_clustering
    current_score = target_func(songs, current_partition, current_W, current_agent_clustering)
    best_score = current_score
    best_W = current_W
    progress_iters = 0  # Track the number of successful iterations
    start_time = time()
    time_elapsed = 0
    for iteration in range(iters):
        T = 1.0 - time_elapsed / max_time  # Annealing schedule
        new_partition, new_W, new_agent_clustering = change_chords_clustering(current_partition, current_W,
                                                                              current_agent_clustering, upper_bound)
        new_score = target_func(songs, new_partition, new_W, new_agent_clustering)
        delta_score = new_score - current_score

        if delta_score < 0 or random.random() < np.exp(-delta_score / T):
            current_partition = new_partition
            current_W = new_W
            current_agent_clustering = new_agent_clustering
            current_score = new_score
            progress_iters = 0

        if delta_score > 0:
            progress_iters += 1

        if new_score < best_score:
            best_W = new_W
            best_score = new_score
        time_elapsed = time() - start_time
        if progress_iters >= success_limit or time_elapsed >= max_time:
            break

    return best_W


def simulated_annealing(songs, W, iters, success_limit, target_func):
    current_W = W
    current_score = target_func(songs, current_W)
    best_score = current_score
    best_W = current_W
    progress_iters = 0  # Track the number of successful iterations
    start_time = time()
    time_elapsed = 0
    for iteration in range(iters):
        T = 1.0 - time_elapsed / max_time  # Annealing schedule
        new_W = change_chords(current_W)
        new_score = target_func(songs, new_W)
        delta_score = new_score - current_score

        if delta_score < 0 or random.random() < np.exp(-delta_score / T):
            current_W = new_W
            current_score = new_score
            progress_iters = 0

        if delta_score > 0:
            progress_iters += 1

        if new_score < best_score:
            best_W = new_W
            best_score = new_score
        time_elapsed = time() - start_time
        if progress_iters >= success_limit or time_elapsed >= max_time:
            break

    return best_W


def kemeny_clustering(songs, iters=max_iters, upper_bound=3):
    W = majority_algorithm(songs)
    upper_bound = min(len(W) - 1, upper_bound)
    partition = [0]
    agent_clustering = [0] * len(songs)
    return simulated_annealing_clustering(songs, partition, W, agent_clustering, iters, upper_bound, iters / 4,
                                          kemeny_clustering_target_func)


def kemeny_clustering_2gram(songs, iters=max_iters, upper_bound=3):
    W = majority_algorithm(songs)
    upper_bound = min(len(W) - 1, upper_bound)
    partition = [0]
    agent_clustering = [0] * len(songs)
    return simulated_annealing_clustering(songs, partition, W, agent_clustering, iters, upper_bound, iters / 4,
                                          kemeny_clustering_2gram_target_func)


def cluster_distance(voters, W, window_size=16):
    s = 0
    for v in voters:
        min_dist = np.inf
        for i in range(len(W) - window_size):
            d = song_distance(v[i:i + window_size], W[i:i + window_size])
            if d < min_dist:
                min_dist = d
        s += min_dist
    return s / len(voters)


def get_variations(song, voters, errors):
    songs = []
    for i in range(voters):
        songs.append(song.copy())
        for j in range(numpy.random.randint(len(song) * errors[0], len(song) * errors[1])):
            place = numpy.random.randint(0, len(song))
            c = songs[-1][place]
            d = chords_distances[c][:c] + [1] + chords_distances[c][c + 1:]
            w = [1 - x for x in d]
            w = [x / sum(w) for x in w]
            songs[-1][place] = random.choices([i for i, x in enumerate(d)], weights=w)[0]
    return songs


if __name__ == '__main__':
    if 0:
        all_songs = load_songs()
        probs = create_2_gram_probability(all_songs)
        algorithms = [majority_algorithm, majority_2gram, kemeny, kemeny_2gram, proportional_algorithm,
                      proportional_2gram_algorithm, kemeny_clustering, kemeny_clustering_2gram]

        errors = [[0, 1], [1, 2], [2, 3], [3, 4]]
        iters = 100
        voters = [8, 16, 32]

        cluster_d = []
        success = []
        sanity = []
        names = []
        for errs in errors:
            for vs in voters:
                names.append('Errors (' + str(errs[0]) + ',' + str(errs[1]) + ') Voters ' + str(vs))
                cluster_d.append([])
                success.append([])
                sanity.append([])
                for i in range(len(algorithms)):
                    success[-1].append([])
                    sanity[-1].append([])
                    cluster_d[-1].append([])
                for i in range(iters):
                    song = get_chords(random.choice(all_songs))
                    songs = get_variations(song, vs, errs)
                    for j in range(len(algorithms)):
                        W = algorithms[j](songs)
                        success[-1][j].append(song_distance(W, song))
                        sanity[-1][j].append(get_chords_probability(W))
                        cluster_d[-1][j].append(cluster_distance(songs, W))
        with open('success.json', 'w') as f:
            json.dump((success, sanity, cluster_d, names), f)
    else:
        with open('success.json', 'r') as f:
            success, sanity, cluster_d, names = json.load(f)
    algorithm_names = ['Majority', 'Majority + 2gram', 'Kemeny', 'Kemeny + 2gram', 'Proportional',
                       'Proportional + 2gram', 'Kemeny Clustering', 'Kemeny Clustering + 2gram']
    res = 'Distance\n'
    for j, s in enumerate(success):
        res += names[j] + ': '
        res += ', '.join(algorithm_names[i] + ' {0:.2g}'.format(x) for i, x in enumerate([sum(x) / len(x) * 100 for x in s])) + '\n'
    res += 'Cluster Distance\n'
    for j, s in enumerate(cluster_d):
        res += names[j] + ': '
        res += ', '.join(algorithm_names[i] + ' {0:.2g}'.format(x) for i, x in enumerate([sum(x) / len(x) * 100 for x in s])) + '\n'
    res += 'Musical Suitability\n'
    for j, s in enumerate(sanity):
        res += names[j] + ': '
        res += ', '.join(algorithm_names[i] + ' {0:.2g}'.format(x) for i, x in enumerate([sum([math.exp(z) for z in x]) / len(x) * 100 for x in s])) + '\n'
    print(res)
    l = 10
    print('Distance')
    s = []
    for y in success:
        s.append([])
        for x in y:
            s[-1].append(x[:l])
    success = s
    for j, s in enumerate(success):
        f = plt.figure()
        x = np.arange(len(s[0]))
        w = 0.9 / len(s)

        for i, y in enumerate(s):
            plt.bar(x + (i - (len(s) - 1) / 2) * w, y, width=w)
        plt.xticks(x, [str(y) for y in x])
        plt.title('Algorithms Distance')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.legend(algorithm_names)
        plt.savefig(names[j] + ' Distance.jpg')
        plt.close(f)
    print('Cluster Distance')
    s = []
    for y in cluster_d:
        s.append([])
        for x in y:
            s[-1].append(x[:l])
    cluster_d = s
    for j, s in enumerate(cluster_d):
        f = plt.figure()
        x = np.arange(len(s[0]))
        for i, y in enumerate(s):
            plt.bar(x + (i - (len(s) - 1) / 2) * w, y, width=w)
        plt.xticks(x, [str(y) for y in x])
        plt.title('Algorithms Cluster Distance')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.legend(algorithm_names)
        plt.savefig(names[j] + ' Cluster Distance.jpg')
        plt.close(f)
    print('Sanity')
    s = []
    for y in sanity:
        s.append([])
        for x in y:
            s[-1].append(x[:l])
    sanity = s
    for j, s in enumerate(sanity):
        f = plt.figure()
        x = np.arange(len(s[0]))
        for i, y in enumerate(s):
            plt.bar(x + (i - (len(s) - 1) / 2) * w, [math.exp(z) for z in y], width=w)
        plt.title('Algorithm Musical suitability')
        plt.xticks(x, [str(y) for y in x])
        plt.xlabel('Iteration')
        plt.ylabel('Musical Suitability')
        plt.legend(algorithm_names)
        plt.savefig(names[j] + ' Sanity.jpg')
        plt.close(f)
