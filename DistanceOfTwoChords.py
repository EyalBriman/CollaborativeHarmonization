chord_dict = {
    'C^': ['C', 'E', 'G'],
    'C#^': ['C#', 'F', 'G#'],
    'D^': ['D', 'G', 'A'],
    'D#^': ['D#', 'G#', 'A#'],
    'E^': ['E', 'G#', 'B'],
    'F^': ['F', 'A', 'C'],
    'F#^': ['F#', 'A#', 'C#'],
    'G^': ['G', 'B', 'D'],
    'G#^': ['G#', 'C', 'D#'],
    'A^': ['A', 'C#', 'E'],
    'A#^': ['A#', 'D', 'F'],
    'B^': ['B', 'D#', 'F#'],

    'C-': ['C', 'D#', 'G'],
    'C#-': ['C#', 'E', 'G#'],
    'D-': ['D', 'F', 'A'],
    'D#-': ['D#', 'F#', 'A#'],
    'E-': ['E', 'G', 'Bb'],
    'F-': ['F', 'G#', 'C'],
    'F#-': ['F#', 'A', 'C#'],
    'G-': ['G', 'A#', 'D'],
    'G#-': ['G#', 'B', 'D#'],
    'A-': ['A', 'C', 'E'],
    'A#-': ['A#', 'C#', 'F'],
    'B-': ['B', 'D', 'F#'],

    'C+': ['C', 'E', 'G#'],
    'C#+': ['C#', 'F', 'A'],
    'D+': ['D', 'F#', 'A#'],
    'D#+': ['D#', 'G#', 'B'],
    'E+': ['E', 'G#', 'C'],
    'F+': ['F', 'A', 'C#'],
    'F#+': ['F#', 'A#', 'D'],
    'G+': ['G', 'B', 'D#'],
    'G#+': ['G#', 'C', 'E'],
    'A+': ['A', 'C#', 'F'],
    'A#+': ['A#', 'D', 'F#'],
    'B+': ['B', 'D#', 'G'],

    'Co': ['C', 'D#', 'G#'],
    'C#o': ['C#', 'E', 'G'],
    'Do': ['D', 'F', 'G#'],
    'D#o': ['D#', 'F#', 'A'],
    'Eo': ['E', 'G', 'A#'],
    'Fo': ['F', 'G#', 'B'],
    'F#o': ['F#', 'A', 'C'],
    'Go': ['G', 'A#', 'C#'],
    'G#o': ['G#', 'B', 'D'],
    'Ao': ['A', 'C', 'D#'],
    'A#o': ['A#', 'C#', 'E'],
    'Bo': ['B', 'D', 'F'],

    'C-7': ['C', 'D#', 'G', 'A#'],
    'C#-7': ['C#', 'E', 'G#', 'B'],
    'D-7': ['D', 'F', 'A', 'C'],
    'D#-7': ['D#', 'F#', 'A#', 'C#'],
    'E-7': ['E', 'G', 'Bb', 'D'],
    'F-7': ['F', 'G#', 'C', 'D#'],
    'F#-7': ['F#', 'A', 'C#', 'E'],
    'G-7': ['G', 'A#', 'D', 'F'],
    'G#-7': ['G#', 'B', 'D#', 'F#'],
    'A-7': ['A', 'C', 'E', 'G'],
    'A#-7': ['A#', 'C#', 'F', 'G#'],
    'B-7': ['B', 'D', 'F#', 'A'],

    'C-7b5': ['C', 'D#', 'G', 'A'],
    'C#-7b5': ['C#', 'E', 'G#', 'Bb'],
    'D-7b5': ['D', 'F', 'A', 'B'],
    'D#-7b5': ['D#', 'F#', 'A#', 'C'],
    'E-7b5': ['E', 'G', 'A#', 'C#'],
    'F-7b5': ['F', 'G#', 'B', 'D'],
    'F#-7b5': ['F#', 'A', 'C', 'D#'],
    'G-7b5': ['G', 'A#', 'C#', 'E'],
    'G#-7b5': ['G#', 'B', 'D', 'F'],
    'A-7b5': ['A', 'C', 'D#', 'F#'],
    'A#-7b5': ['A#', 'C#', 'E', 'G'],
    'B-7b5': ['B', 'D', 'F', 'G#'],

    'C^7': ['C', 'E', 'G', 'Bb'],
    'C#^7': ['C#', 'F', 'G#', 'B'],
    'D^7': ['D', 'G', 'A', 'C'],
    'D#^7': ['D#', 'G#', 'A#', 'C#'],
    'E^7': ['E', 'G#', 'B', 'D'],
    'F^7': ['F', 'A', 'C', 'D#'],
    'F#^7': ['F#', 'A#', 'C#', 'E'],
    'G^7': ['G', 'B', 'D', 'F'],
    'G#^7': ['G#', 'C', 'D#', 'F#'],
    'A^7': ['A', 'C#', 'E', 'G'],
    'A#^7': ['A#', 'D', 'F', 'G#'],
    'B^7': ['B', 'D#', 'F#', 'A'],

    'C-^7': ['C', 'D#', 'G#', 'Bb'],
    'C#-^7': ['C#', 'E', 'G', 'B'],
    'D-^7': ['D', 'F', 'A', 'C'],
    'D#-^7': ['D#', 'F#', 'A#', 'C#'],
    'E-^7': ['E', 'G', 'Bb', 'D'],
    'F-^7': ['F', 'G#', 'C', 'D#'],
    'F#-^7': ['F#', 'A', 'C#', 'E'],
    'G-^7': ['G', 'A#', 'D', 'F'],
    'G#-^7': ['G#', 'B', 'D#', 'F#'],
    'A-^7': ['A', 'C', 'E', 'G'],
    'A#-^7': ['A#', 'C#', 'F', 'G#'],
    'B-^7': ['B', 'D', 'F#', 'A'],

    'Co7': ['C', 'D#', 'G#', 'A'],
    'C#o7': ['C#', 'E', 'G', 'A#'],
    'Do7': ['D', 'F', 'G#', 'B'],
    'D#o7': ['D#', 'F#', 'A', 'C'],
    'Eo7': ['E', 'G', 'A#', 'C#'],
    'Fo7': ['F', 'G#', 'B', 'D'],
    'F#o7': ['F#', 'A', 'C', 'D#'],
    'Go7': ['G', 'A#', 'D', 'E'],
    'G#o7': ['G#', 'B', 'D#', 'F'],
    'Ao7': ['A', 'C', 'D#', 'F#'],
    'A#o7': ['A#', 'C#', 'E', 'G'],
    'Bo7': ['B', 'D', 'F', 'G#'],
     'C7': ['C', 'E', 'G', 'A#'],
    'C#7': ['C#', 'F', 'G#', 'B'],
    'D7': ['D', 'G', 'A', 'C'],
    'D#7': ['D#', 'G#', 'A#', 'C#'],
    'E7': ['E', 'G#', 'B', 'D'],
    'F7': ['F', 'A', 'C', 'D#'],
    'F#7': ['F#', 'A#', 'C#', 'E'],
    'G7': ['G', 'B', 'D', 'F'],
    'G#7': ['G#', 'C', 'D#', 'F#'],
    'A7': ['A', 'C#', 'E', 'G'],
    'A#7': ['A#', 'D', 'F', 'G#'],
    'B7': ['B', 'D#', 'F#', 'A'],

    'C-6': ['C', 'D#', 'G', 'A'],
    'C#-6': ['C#', 'E', 'G#', 'A#'],
    'D-6': ['D', 'F', 'A', 'B'],
    'D#-6': ['D#', 'F#', 'A#', 'C'],
    'E-6': ['E', 'G', 'A#', 'C#'],
    'F-6': ['F', 'G#', 'B', 'D'],
    'F#-6': ['F#', 'A', 'C', 'D#'],
    'G-6': ['G', 'A#', 'D', 'E'],
    'G#-6': ['G#', 'B', 'D#', 'F'],
    'A-6': ['A', 'C', 'D#', 'F#'],
    'A#-6': ['A#', 'C#', 'E', 'G'],
    'B-6': ['B', 'D', 'F', 'G#'],

    'C^6': ['C', 'E', 'G', 'A'],
    'C#^6': ['C#', 'F', 'G#', 'A#'],
    'D^6': ['D', 'G', 'A', 'B'],
    'D#^6': ['D#', 'G#', 'A#', 'C'],
    'E^6': ['E', 'G#', 'B', 'C#'],
    'F^6': ['F', 'A', 'C', 'D'],
    'F#^6': ['F#', 'A#', 'C#', 'D#'],
    'G^6': ['G', 'B', 'D', 'E'],
    'G#^6': ['G#', 'C', 'D#', 'F'],
    'A^6': ['A', 'C#', 'E', 'F#'],
    'A#^6': ['A#', 'D', 'F', 'G'],
    'B^6': ['B', 'D#', 'F#', 'G#'],
}
# Define the 12 half tones
half_tones = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Define the 5 triads
triads = ['^', '-', '+', 'o']

# Define the 6 7th chords
seventh_chords = ['-7', '-7b5', '^7', '-^7', 'o7', '7']

# Define the 2 6th chords
sixth_chords = ['-6', '^6']

# Generate all 144 chord combinations
all_chords = []
for tone in half_tones:
    for triad in triads:
        all_chords.append(tone + triad)
    for seventh in seventh_chords:
        all_chords.append(tone + seventh)
    for sixth in sixth_chords:
        all_chords.append(tone + sixth)

# Map each chord to a number ranging from 0 to 144
chord_to_number = {chord: i+1  for i, chord in enumerate(all_chords)}
def distances(chord1,chord2):
    if (abs(chord_to_number[chord1]-chord_to_number[chord2]) ==0):
        return 0
    elif (abs(chord_to_number[chord1]-chord_to_number[chord2]) in [3,103,109,106,38,112,107,35,32,46,52,49,43,5,4,11,44,47,35]):
        if ((chord1[-1]==chord2[-1] and chord1[-1]=='7') or len(chord1)==len(chord2)):
            return 0.5
        else:
            return 1
    elif (abs(chord_to_number[chord1]-chord_to_number[chord2]) in [1,2,10]):
        return 2
    elif (abs(chord_to_number[chord1]-chord_to_number[chord2]) in [96,36,48,108,114,102]):
        return 3
    elif (abs(chord_to_number[chord1]-chord_to_number[chord2]) in [12,132]):
        return 4
    else:
        set1 = set(chord_dict[chord1])
        set2 = set(chord_dict[chord2])

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        jaccard_distance = 1 - intersection / union
        return 4+jaccard_distance
