import scipy
import random
import time
import numpy as np
from two_gram import *
import gurobipy as gp
from gurobipy import GRB

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


def get_majority_score(majority, W):
    return sum([majority[x, i] for i, x in enumerate(W)]) / len(W)


def proportional_target_func(W, songs):
    res = 0
    for i in range(len(songs)):
        dist = np.sort([chord_distance_quick(int(W[j]), songs[i][j]) for j in range(len(W))])
        res += sum([dist[j] / (j + 1) for j in range(len(dist))])
    return res


def proportional_2gram_target_func(W, songs):
    score1 = -get_2gram_score(get_ints(W))
    score2 = proportional_target_func(W, songs)
    return score1 + score2


def majority_2gram_target_func(W, majority):
    Wi = get_ints(W)
    score1 = -get_2gram_score(Wi)
    score2 = -get_majority_score(majority, Wi) * 50
    return score1 + score2


def kemeny_target_func(W, songs):
    Wi = get_ints(W)
    return sum([sum([chord_distance(Wi[i], song[i]) for i in range(len(song))]) for song in songs])


def kemeny_2gram_target_func(W, songs):
    Wi = get_ints(W)
    score1 = -get_2gram_score(Wi)
    score2 = sum([sum([chord_distance(Wi[i], song[i]) for i in range(len(song))]) for song in songs]) / len(
        songs) / len(songs[0]) * 4
    return score1 + score2


def change_chords(W):
    place = numpy.random.randint(0, len(W))
    W[place] = (W[place] + numpy.random.randint(1, 144)) % 144
    return W


def majority_algorithm(songs):
    song_len = len(songs[0])
    majority = numpy.zeros((144, song_len))

    for song in songs:
        for i, chord in enumerate(song):
            majority[chord, i] += 1
    return list(numpy.argmax(majority, axis=0))


def proportional_algorithm(songs, iters=5000, init_with_majority=True):
    if init_with_majority:
        init = majority_algorithm(songs)
    else:
        init = numpy.random.randint(0, 144, len(songs[0]))
    return get_ints(scipy.optimize.basinhopping(proportional_target_func, init, niter=iters, niter_success=750,
                                                take_step=change_chords, minimizer_kwargs={'args': songs}).x)


def kemeny(songs, iters=5000, init_with_majority=True):
    if init_with_majority:
        init = majority_algorithm(songs)
    else:
        init = numpy.random.randint(0, 144, len(songs[0]))
    return get_ints(scipy.optimize.basinhopping(kemeny_target_func, init, niter=iters, niter_success=750,
                                                take_step=change_chords, minimizer_kwargs={'args': songs}).x)


def proportional_2gram_algorithm(songs, iters=5000, init_with_majority=True):
    if init_with_majority:
        init = majority_algorithm(songs)
    else:
        init = numpy.random.randint(0, 144, len(songs[0]))
    return get_ints(scipy.optimize.basinhopping(proportional_2gram_target_func, init, niter=iters, niter_success=750,
                                                take_step=change_chords, minimizer_kwargs={'args': songs}).x)


def kemeny_2gram(songs, iters=5000, init_with_majority=True):
    if init_with_majority:
        init = majority_algorithm(songs)
    else:
        init = numpy.random.randint(0, 144, len(songs[0]))
    return get_ints(scipy.optimize.basinhopping(kemeny_2gram_target_func, init, niter=iters, niter_success=750,
                                                take_step=change_chords, minimizer_kwargs={'args': songs}).x)


def majority_2gram_algorithm(songs, iters=5000, init_with_majority=True):
    song_len = len(songs[0])
    majority = numpy.zeros((144, song_len))

    for song in songs:
        for i, chord in enumerate(song):
            majority[chord, i] += 1 / len(songs)

    if init_with_majority:
        init = numpy.argmax(majority, axis=0)
    else:
        init = numpy.random.randint(0, 144, song_len)
    return get_ints(scipy.optimize.basinhopping(majority_2gram_target_func, init, niter=iters, niter_success=750,
                                                take_step=change_chords, minimizer_kwargs={'args': majority}).x)


def P(j, z, Z):
    if (z < len(Z) - 1 and j >= Z[z] and z < Z[z + 1]) or (z == len(Z) - 1 and j >= Z[z]):
        return True
    return False


def kemeny_clustering_algorithm(songs, Z):
    Z.sort()
    if len(songs) == 0 or Z[0] != 0 or Z[-1] > len(songs[0]):
        raise ValueError('Songs must not be empty and Z must start with 0 and not exceed length of songs')
    input = []
    for i in range(len(songs)):
        input.append([])
        for j in range(len(songs[0])):
            input[i].append([songs[i][j] // 12, songs[i][j] % 12])
    m = gp.Model()
    a = m.addVars(len(songs), vtype=GRB.INTEGER, lb=0, ub=len(Z) - 1, name="a")
    Q = m.addVars(len(songs), len(Z), vtype=GRB.BINARY, name="Q")
    W = m.addVars(len(songs[0]), 2, vtype=GRB.INTEGER, name="W")
    T = m.addVars(len(songs), len(songs[0]), 2, vtype=GRB.INTEGER, ub=11, lb=0, name="T")
    T2 = m.addVars(len(songs), len(songs[0]), 2, vtype=GRB.INTEGER, ub=11, lb=0, name="T2")
    obj = 0
    for i in range(len(songs)):
        for z in range(len(Z)):
            for j in range(len(songs[0])):
                if P(j, z, Z):
                    obj += (Q[i, z] * 0.7 + 0.3) * ((T[i, j, 0] + T[i, j, 1]) * 4 + T2[i, j, 0] + T2[i, j, 1])
    m.setObjective(obj, GRB.MINIMIZE)
    for i in range(len(songs)):
        for z in range(len(Z)):
            m.addConstr((Q[i, z] == 1) >> (a[i] == z), name="C0" + str(i) + str(z))
        m.addConstr(gp.quicksum([Q[i, z] for z in range(len(Z))]) == 1, name="C1" + str(i))
    for i in range(len(songs)):
        for j in range(len(songs[0])):
            m.addConstr(T[i, j, 0] - T[i, j, 1] == W[j, 0] - input[i][j][0], name="C2" + str(i) + str(j))
            m.addConstr(T2[i, j, 0] - T2[i, j, 1] == W[j, 1] - input[i][j][1], name="C3" + str(i) + str(j))
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return []
    res = []
    for v in m.getVars():
        if v.VarName[0] == 'W':
            if v.VarName[4] == '0':
                res.append(int(v.X) * 12)
            else:
                res[-1] += int(v.X)
        elif v.VarName[0] == 'a' or v.VarName[0] == 'Q':
            print(v.VarName, int(v.X))
    return res


def get_variations(song, n):
    songs = []
    for i in range(n):
        songs.append(song.copy())
        for j in range(numpy.random.randint(1, round(len(song)))):
            place = numpy.random.randint(0, len(song))
            c = songs[-1][place]
            d = chord_distances[c * 144:c * 144 + c] + chord_distances[c * 144 + c + 1:c * 144 + 144]
            songs[-1][place] = numpy.random.choice([i for i, x in enumerate(d) if x == min(d)])
    return songs


if __name__ == '__main__':
    all_songs = load_songs()
    probs = create_2_gram_probability(all_songs)
    algorithms = [majority_algorithm, kemeny, proportional_algorithm, majority_2gram_algorithm, kemeny_2gram,
                  proportional_2gram_algorithm]
    success = [0] * len(algorithms)
    iters = 500
    voters = 32
    t_0 = time.time()
    for i in range(iters):
        song = get_chords(random.choice(all_songs))
        songs = get_variations(song, voters)
        for j in range(len(algorithms)):
            if algorithms[j](songs) == song:
                success[j] += 1
    t_1 = time.time()
    print(t_1 - t_0)
    print([x / iters for x in success])