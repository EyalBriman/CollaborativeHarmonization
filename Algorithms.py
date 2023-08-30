import scipy
import random
from two_gram import *


def get_chords_probability(chords):
    x = []
    for i in range(len(chords) - 1):
        try:
            x.append(probs[chords[i]][chords[i + 1]])
        except KeyError:
            x.append(lepsilon)
    return combine_probabilities(x)


def get_ints(x):
    return [int(z) for z in x]


def get_2gram_score(song):
    return get_chords_probability(song)


def get_majority_score(majority, best_song):
    return sum([majority[x, i] for i, x in enumerate(best_song)]) / len(best_song)


def majority_2gram_target_func(best_song, majority):
    x = get_ints(best_song)
    s1 = -get_2gram_score(x)
    s2 = -get_majority_score(majority, x) * 50
    return s1 + s2


def kemeny_target_func(best_song, songs):
    x = get_ints(best_song)
    return sum([sum([chord_distance(x[i], song[i]) for i in range(len(song))]) for song in songs])


def kemeny_2gram_target_func(best_song, songs):
    x = get_ints(best_song)
    s1 = -get_2gram_score(x)
    s2 = sum([sum([chord_distance(x[i], song[i]) for i in range(len(song))]) for song in songs]) / len(songs) / len(
        songs[0]) * 4
    return s1 + s2


def change_chords(x):
    place = numpy.random.randint(0, len(x))
    x[place] = (x[place] + numpy.random.randint(1, 144)) % 144
    return x


def majority_algorithm(songs):
    song_len = len(songs[0])
    majority = numpy.zeros((144, song_len))

    for song in songs:
        for i, chord in enumerate(song):
            majority[chord, i] += 1
    return list(numpy.argmax(majority, axis=0))


def kemeny(songs, iters=5000, init_with_majority=True):
    if init_with_majority:
        init = majority_algorithm(songs)
    else:
        init = numpy.random.randint(0, 144, len(songs[0]))
    return get_ints(scipy.optimize.basinhopping(kemeny_target_func, init, niter=iters, niter_success=750,
                                                take_step=change_chords, minimizer_kwargs={'args': songs}).x)


def kemeny_2gram(songs, iters=5000, init_with_majority=True):
    if init_with_majority:
        init = majority_algorithm(songs)
    else:
        init = numpy.random.randint(0, 144, len(songs[0]))
    return get_ints(scipy.optimize.basinhopping(kemeny_2gram_target_func, init, niter=iters, niter_success=750,
                                                take_step=change_chords, minimizer_kwargs={'args': songs}).x)


def majority_2gram_algorithm(songs, iters=250000, init_with_majority=True):
    song_len = len(songs[0])
    majority = numpy.zeros((144, song_len))

    for song in songs:
        for i, chord in enumerate(song):
            majority[chord, i] += 1 / len(songs)

    if init_with_majority:
        init = numpy.argmax(majority, axis=0)
    else:
        init = numpy.random.randint(0, 144, song_len)
    return get_ints(scipy.optimize.basinhopping(majority_2gram_target_func, init, niter=iters, niter_success=50000,
                                                take_step=change_chords, minimizer_kwargs={'args': majority}).x)


if __name__ == '__main__':
    all_songs = load_songs()
    probs = create_2_gram_probability(all_songs)
    while True:
        song = get_chords(random.choice(all_songs))
        songs = []
        for i in range(16):
            songs.append(song.copy())
            for j in range(round(len(song))):
                place = numpy.random.randint(0, len(song))
                songs[-1][place] = (songs[-1][place] + numpy.random.choice(
                    [3, 103, 109, 106, 38, 112, 107, 35, 32, 46, 52, 49, 43, 5, 4, 11, 44, 47, 35])) % 144
        y = majority_algorithm(songs)
        if y != song:
            break
    print(song)
    print(y)
    if 0:
        y = majority_2gram_algorithm(songs)
        print(y)
        print(y == song)
    y = kemeny(songs)
    print(y)
    print(y == song)
