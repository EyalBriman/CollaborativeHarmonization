import math
import matplotlib.pyplot as plt
import random
import scipy
import numpy as np
from two_gram import *
# import gurobipy as gp
# from gurobipy import GRB
from DistanceOfTwoChords import chords_distances


def change_chords(W):
    place = numpy.random.randint(0, len(W))
    W[place] = (W[place] + numpy.random.randint(1, round(change_chords.stepsize))) % len(chords_distances)
    return W


def change_chords_clustering(args):
    flag = random.random()
    if flag <= 0.4:
        place = numpy.random.randint(0, len(args['W']))
        args['W'][place] = (args['W'][place] + numpy.random.randint(1, len(chords_distances))) % len(chords_distances)
    elif 0.4 < flag <= 0.8:
        flag_1 = numpy.random.randint(0, len(args['agent_clustering']))
        args['agent_clustering'][flag_1] = random.choice(args['partition'])
    elif 0.8 < flag <= 0.9:
        flag_2 = len(W['partition'])
        args['partition'] = random.sample(range(len(args['W'])), flag_2)
    else:
        flag_3 = numpy.random.randint(0, change_chords_clustering.upper_bound + 1)
        args['partition'] = random.sample(range(len(W)), flag_3)
    return args


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


def kemeny_clustering(songs, upper_bound, iters=5000):
    W = majority_algorithm(songs)
    args = {}
    args['W'] = W
    flag_3 = numpy.random.randint(0, upper_bound + 1)
    args['partition'] = random.sample(range(len(W)), flag_3)
    args['agent_clustering'] = []
    for i in range(len(songs)):
        args['agent_clustering'].append(random.choice(args['partition']))
    change_chords_clustering.upper_bound = upper_bound
    return get_ints(scipy.optimize.basinhopping(kemeny_clustering_target_func, args, niter=iters, niter_success=750,
                                                take_step=change_chords_clustering, minimizer_kwargs={'args': songs}).x[
                        'W'])

    def get_variations(song, n):
        songs = []
        for i in range(n):
            songs.append(song.copy())
            for j in range(numpy.random.randint(len(songs), len(song) * 4)):
                place = numpy.random.randint(0, len(song))
                c = songs[-1][place]
                d = chords_distances[c][:c] + [1] + chords_distances[c][c + 1:]
                songs[-1][place] = numpy.random.choice([i for i, x in enumerate(d) if x <= max(0.5, min(d))])
        return songs

    if __name__ == '__main__':
        if 0:
            all_songs = load_songs()
            probs = create_2_gram_probability(all_songs)
            algorithms = [majority_algorithm, majority_2gram, kemeny, kemeny_2gram, proportional_algorithm,
                          proportional_2gram_algorithm]
            success = []
            sanity = []
            for i in range(len(algorithms)):
                success.append([])
                sanity.append([])

            iters = 5
            voters = 8
            for i in range(iters):
                song = get_chords(random.choice(all_songs))
                songs = get_variations(song, voters)
                for j in range(len(algorithms)):
                    W = algorithms[j](songs)
                    success[j].append(song_distance(W, song))
                    sanity[j].append(get_chords_probability(W))
            with open(
                    'C:\\Users\\Eleizerovich\\OneDrive - Gita Technologies LTD\\Desktop\\School\\CollaborativeHarmonization\\success.json',
                    'w') as f:
                json.dump((success, sanity), f)
        else:
            with open(
                    'C:\\Users\\Eleizerovich\\OneDrive - Gita Technologies LTD\\Desktop\\School\\CollaborativeHarmonization\\success.json',
                    'r') as f:
                (success, sanity) = json.load(f)
        print([sum(x) / len(x) * 100 for x in success])
        x = np.arange(len(success[0]))
        w = 0.8 / len(success)
        for i, y in enumerate(success):
            plt.bar(x + (i - len(success) / 2) * w, y, width=w)
        plt.xticks(x, [str(y) for y in x])
        plt.title('Algorithms Distance')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.legend(['Majority', 'Majority-2gram', 'Kemeny', 'Kemeny-2gram', 'Proportional', 'Proportional-2gram'])
        plt.show()
        for i, y in enumerate(sanity):
            plt.bar(x + (i - len(sanity) / 2) * w, [math.exp(z) for z in y], width=w)
        plt.title('Algorithm Musical suitability')
        plt.xlabel('Iteration')
        plt.ylabel('Musical Suitability')
        plt.legend(['Majority', 'Majority-2gram', 'Kemeny', 'Kemeny-2gram', 'Proportional', 'Proportional-2gram'])
        plt.show()
