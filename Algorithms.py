import math
import matplotlib.pyplot as plt
import random
import scipy
import numpy as np
from two_gram import *
from DistanceOfTwoChords import chords_distances


def change_chords(W):
    place = numpy.random.randint(0, len(W))
    W[place] = (W[place] + numpy.random.randint(1, round(change_chords.stepsize))) % len(chords_distances)
    return W


def change_chords_clustering(partition, W, agent_clustering, upper_bound=4):
    flag = random.random()
    W = W.copy()
    partition = partition.copy()
    agent_clustering = agent_clustering.copy()
    if flag <= 0.4:
        place = np.random.randint(0, len(W))
        W[place] = (W[place] + np.random.randint(1, 120)) % 120
    elif flag > 0.4 and flag <= 0.8:
        flag_1 = np.random.randint(0, len(agent_clustering))
        agent_clustering[flag_1] = random.choice(partition)
    elif flag > 0.8 and flag <= 0.9:
        flag_2 = len(partition)
        partition = [0] + random.sample(range(1, len(W)), flag_2)
    else:
        flag_3 = np.random.randint(0, upper_bound + 1)
        partition = [0] + random.sample(range(1, len(W)), flag_3)
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


def proportional_target_func(W, songs):
    res = 0
    W = get_ints(W)
    for i in range(len(songs)):
        dist = np.sort([chords_distances[W[j]][songs[i][j]] for j in range(len(W))])
        res += sum([dist[j] / (j + 1) for j in range(len(dist))])
    return res


def proportional_2gram_target_func(W, songs):
    score1 = -get_2gram_score(get_ints(W))
    score2 = proportional_target_func(W, songs)
    return score1 + score2


def majority_algorithm(songs):
    song_len = len(songs[0])
    majority = numpy.zeros((len(chords_distances), song_len))

    for song in songs:
        for i, chord in enumerate(song):
            majority[chord, i] += 1
    return list(numpy.argmax(majority, axis=0))


def proportional_algorithm(songs, iters=5000, init_with_majority=True):
    if init_with_majority:
        init = majority_algorithm(songs)
    else:
        init = numpy.random.randint(0, len(chords_distances), len(songs[0]))
    change_chords.stepsize = 20
    return get_ints(scipy.optimize.basinhopping(proportional_target_func, init, niter=iters, niter_success=750,
                                                take_step=change_chords, minimizer_kwargs={'args': songs}).x)


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


def proportional_2gram_algorithm(songs, iters=5000, init_with_majority=True):
    if init_with_majority:
        init = majority_algorithm(songs)
    else:
        init = numpy.random.randint(0, len(chords_distances), len(songs[0]))
    change_chords.stepsize = 20
    return get_ints(
        scipy.optimize.basinhopping(proportional_2gram_target_func, init, niter=iters, niter_success=750,
                                    take_step=change_chords, minimizer_kwargs={'args': songs}).x)


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
                    d = base_T[j] + T[m] + probs[m][j]
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
                base_T[song[i]] -= 1
        if i > 0:
            new_T = []
            min_d = math.inf
            for j in range(len(chords_distances)):
                for m in range(len(chords_distances)):
                    d = base_T[j] + T[m] + probs[m][j]
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


def simulated_annealing(songs, partition, W, agent_clustering, iters, upper_bound, success_limit):
    current_partition = partition
    current_W = W
    current_agent_clustering = agent_clustering
    current_score = kemeny_clustering_target_func(songs, current_partition, current_W, current_agent_clustering)
    best_partition = current_partition
    best_score = current_score
    best_W = current_W
    best_agent_clustering = current_agent_clustering
    progress_iters = 0  # Track the number of successful iterations

    for iteration in range(iters):
        T = 1.0 - iteration / iters  # Annealing schedule
        new_partition, new_W, new_agent_clustering = change_chords_clustering(current_partition, current_W,
                                                                              current_agent_clustering, upper_bound)
        new_score = kemeny_clustering_target_func(songs, new_partition, new_W, new_agent_clustering)

        delta_score = new_score - current_score

        if delta_score < 0 or random.random() < np.exp(-delta_score / T):
            current_partition = new_partition
            current_W = new_W
            current_agent_clustering = new_agent_clustering
            current_score = new_score
            progress_iters = 0

        if delta_score < 0:
            progress_iters += 1

        if new_score < best_score:
            best_partition = new_partition
            best_W = new_W
            best_agent_clustering = new_agent_clustering
            best_score = new_score

        if progress_iters >= success_limit:
            break

    return best_partition, best_W, best_agent_clustering, best_score


def kemeny_clustering(songs, iters=5000, upper_bound=4, success_limit=750):
    W = majority_algorithm(songs)
    upper_bound = min(len(W) - 1, upper_bound)
    partition = [0]
    agent_clustering = [0] * len(songs)
    best_partition, best_W, best_agent_clustering, best_score = simulated_annealing(songs, partition, W,
                                                                                    agent_clustering, iters,
                                                                                    upper_bound, success_limit)
    return best_W


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
    if 1:
        all_songs = load_songs()
        probs = create_2_gram_probability(all_songs)
        algorithms = [majority_algorithm, majority_2gram, kemeny, kemeny_2gram, proportional_algorithm,
                      proportional_2gram_algorithm, kemeny_clustering]
        errors = [[0, 1], [1, 2], [2, 3], [3, 4]]
        iters = 1
        voters = [8, 16, 32]
        success = []
        sanity = []
        names = []
        for errs in errors:
            for vs in voters:
                names.append('Errors (' + str(errs[0]) + ',' + str(errs[1]) + ') Voters ' + str(vs))
                success.append([])
                sanity.append([])
                for i in range(len(algorithms)):
                    success[-1].append([])
                    sanity[-1].append([])
                for i in range(iters):
                    song = get_chords(random.choice(all_songs))
                    songs = get_variations(song, vs, errs)
                    for j in range(len(algorithms)):
                        W = algorithms[j](songs)
                        success[-1][j].append(song_distance(W, song))
                        sanity[-1][j].append(get_chords_probability(W))
        with open('success.json', 'w') as f:
            json.dump((success, sanity, names), f)
    else:
        with open('success.json', 'r') as f:
            success, sanity, names = json.load(f)
    algorithm_names = ['Majority', 'Majority-2gram', 'Kemeny', 'Kemeny-2gram', 'Proportional', 'Proportional-2gram',
                       'Kemeny Clustering']
    print('Distance')
    for j, s in enumerate(success):
        print(names[j])
        print([sum(x) / len(x) * 100 for x in s])
        x = np.arange(len(s[0]))
        w = 0.8 / len(s)

        for i, y in enumerate(s):
            plt.bar(x + (i - len(s) / 2) * w, y, width=w)
        plt.xticks(x, [str(y) for y in x])
        plt.title('Algorithms Distance')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.legend(algorithm_names)
    print('Sanity')
    for j, s in enumerate(sanity):
        print(names[j])
        print([sum([math.exp(z) for z in x]) / len(x) * 100 for x in s])
        x = np.arange(len(s[0]))
        w = 0.8 / len(s)
        for i, y in enumerate(s):
            plt.bar(x + (i - len(s) / 2) * w, [math.exp(z) for z in y], width=w)
        plt.title('Algorithm Musical suitability')
        plt.xticks(x, [str(y) for y in x])
        plt.xlabel('Iteration')
        plt.ylabel('Musical Suitability')
        plt.legend(algorithm_names)
