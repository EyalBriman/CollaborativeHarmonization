import json
import numpy

all_valid_qualities = {'': '^', '5': '^', '2': '^', 'add9': '^', '+': '+', 'o': 'o', 'h': 'o', 'sus': '^', '^': '^',
                       '-': '-', '^7': '^7', '-7': '-7', '7': '7', '7sus': '7', 'h7': '-7b5', 'o7': 'o7', 'o^7': 'o7',
                       '^9': '^7', '^13': '^7', '6': '^6', '69': '^6', '^7#11': '^7', '^9#11': '^7', '^7#5': '+^7',
                       '-6': '-6', '-69': '-6', '-^7': '-^7', '-^9': '-^7', '-9': '-7', '-11': '-7', '-7b5': '-7b5',
                       'h9': '-7b5;', '-b6': '-', '-#5': '-', '9': '7', '7b9': '7', '7#9': '7', '7#11': '7', '7b5': '7',
                       '7#5': '7', '9#11': '7', '9b5': '7', '9#5': '7', '7b13': '7', '7#9#5': '7', '7#9b5': '7',
                       '7#9#11': '7', '7b9#11': '7', '7b9b5': '7', '7b9#5': '7', '7b9#9': '7', '7b9b13': '7',
                       '7alt': '7', '13': '7', '13#11': '7', '13b9': '7', '13#9': '7', '7b9sus': '7', '7susadd3': '7',
                       '9sus': '7', '13sus': '7', '7b13sus': '7', '11': '7'}

all_pitches = {'C': 0, 'C#': 1, 'Cb': 11, 'D': 2, 'D#': 3, 'Db': 1, 'E': 4, 'E#': 5, 'Eb': 3, 'F': 5, 'F#': 6, 'Fb': 4,
               'G': 7, 'G#': 8, 'Gb': 6, 'A': 9, 'A#': 10, 'Ab': 8, 'B': 11, 'B#': 0, 'Bb': 10}

all_pitches_reverse = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A',
                       10: 'A#', 11: 'B'}

all_valid_chords = None
epsilon = 1e-8
lepsilon = numpy.log(epsilon)


def back_translate_chord(pair):
    return all_pitches_reverse[pair[0]] + pair[1]


def translate_chord(chord):
    if not chord:
        return None
    global all_valid_chords, all_pitches, all_valid_qualities
    if not all_valid_chords:
        all_valid_chords = {}
        for pitch in all_pitches.keys():
            for quality in all_valid_qualities.keys():
                all_valid_chords[pitch + quality] = (all_pitches[pitch], all_valid_qualities[quality])
                for bass_pitch in all_pitches.keys():
                    all_valid_chords[pitch + quality + '/' + bass_pitch] = (
                        all_pitches[pitch], all_valid_qualities[quality])
    if chord not in all_valid_chords:
        raise ValueError('Unknown Chord ' + chord)
    return all_valid_chords[chord]


def get_chords(song):
    chords = []
    for measure in song:
        chords += filter(lambda x: x, [translate_chord(chord) for chord in measure])
    return chords


def create_2_gram_probability(songs, distance=1):
    chord_counts = {}
    chord_followers = {}

    for song in songs:
        chords = get_chords(song)
        for chord_diff in range(12):
            for i in range(len(chords)):
                chords[i] = ((chords[i][0] + 1) % 12, chords[i][1])
            for i in range(len(chords) - distance):
                current_chord = chords[i]
                next_chord = chords[i + distance]

                if current_chord not in chord_counts:
                    chord_counts[current_chord] = 0
                    chord_followers[current_chord] = {}

                chord_counts[current_chord] += 1

                if next_chord not in chord_followers[current_chord]:
                    chord_followers[current_chord][next_chord] = 0

                chord_followers[current_chord][next_chord] += 1

    # Calculate probabilities
    chord_probabilities = {}
    for chord, count in chord_counts.items():
        chord_probabilities[chord] = {}
        for follower, follower_count in chord_followers[chord].items():
            chord_probabilities[chord][follower] = numpy.log(follower_count / count + epsilon)

    return chord_probabilities


def combine_probabilities(x):
    return sum(x) / len(x)


def get_song_probability(song, probs):
    chords = get_chords(song)
    x = []
    for i in range(len(chords) - 1):
        y = []
        for d in range(len(probs)):
            try:
                if i + d + 1 < len(chords):
                    y.append(probs[d][chords[i]][chords[i + d + 1]])
            except KeyError:
                y.append(lepsilon)
        x.append(combine_probabilities(y))
    return combine_probabilities(x)


def run_k_fold(songs, k=1, distance=1):
    x = []
    for i in range(0, len(songs), k):
        ik = min(i + k, len(songs))
        probs = []
        for d in range(1, distance + 1):
            probs.append(create_2_gram_probability(songs[:i] + songs[ik:], d))
        x.append(combine_probabilities([get_song_probability(song, probs) for song in songs[i:ik]]))
    return combine_probabilities(x)


def print_log(chord_probabilities):
    # Example output for the final 2-gram model
    for chord, followers in chord_probabilities.items():
        print(f"Chord: {back_translate_chord(chord)}")
        for follower, lprobability in followers.items():
            print(f"  -> {back_translate_chord(follower)}: Probability: {numpy.exp(lprobability):.3f}")


def test_grams(songs, n=3, k=10):
    for i in range(1, n + 1):
        print('Distance up to ', n, ' log probability:', run_k_fold(songs, k, i))


def load_songs():
    with open('songs.json', "r") as file:
        return json.load(file)


if __name__ == '__main__':
    s = load_songs()
    print_log(create_2_gram_probability(s))
    test_grams(s)
