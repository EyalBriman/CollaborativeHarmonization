# Define the 12 half tones
half_tones = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Define the 5 triads
triads = ['^', '-', '+', 'o']

# Define the 6 7th chords
seventh_chords = ['-7', '-7b5', '^7', '-^7', 'o7', 'd7']

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
        if ((chord1[-1]==chord2[-1] and chord1[-1]=='7') or len(chord1)==chord2):
            return 0.5
        else:
            return 1
    elif (abs(chord_to_number[chord1]-chord_to_number[chord2]) in [1,2,10]):
        return 3
    elif (abs(chord_to_number[chord1]-chord_to_number[chord2]) in [96,36,48,108,114,102]):
        return 5
    elif (abs(chord_to_number[chord1]-chord_to_number[chord2]) in [12,132]):
        return 6
