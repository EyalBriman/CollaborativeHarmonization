half_tones = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
chords = {'^7': [0, 4, 7, 11], '-7': [0, 3, 7, 10], '-6': [0, 3, 7, 9], '-^7': [0, 3, 7, 11], '+^7': [0, 4, 8, 11],
          '+7': [0, 4, 8, 10], '7': [0, 4, 7, 10],
          '-7b5': [0, 3, 6, 10], 'o': [0, 3, 6, 9], 'o^7': [0, 3, 6, 11]}
all_chords = []
chord_dict = {}
for i, tone in enumerate(half_tones):
    for j, chord in enumerate(list(chords.keys())):
        all_chords.append(tone + chord)
        chord_dict[tone + chord] = []
        for note in chords[chord]:
            chord_dict[tone + chord].append(half_tones[(note + i)%len(half_tones)])

# Map each chord to a number ranging from 0 to 144
chord_to_number = {chord: i for i, chord in enumerate(all_chords)}
number_to_chord = {i: chord for i, chord in enumerate(all_chords)}


def chord_distance(chord1, chord2):
    set1 = set(chord_dict[number_to_chord[chord1]])
    set2 = set(chord_dict[number_to_chord[chord2]])

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return 1 - intersection / union


def init_distances():
    global chords_distances
    for chord1 in range(len(all_chords)):
        chords_distances.append([])
        for chord2 in range(len(all_chords)):
            chords_distances[-1].append(chord_distance(chord1, chord2))


chords_distances = []
init_distances()
