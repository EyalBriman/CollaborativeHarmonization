import scipy
import random
from two_gram import *
import gurobipy as gp
from gurobipy import GRB

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

###distances
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
chord_to_number = {chord: i for i, chord in enumerate(all_chords)}
number_to_chord = {i: chord for i, chord in enumerate(all_chords)}


def chord_distance(chord1, chord2):
    d = abs(chord1 - chord2)
    if d == 0:
        return 0
    elif d in [3, 103, 109, 106, 38, 112, 107, 35, 32, 46, 52, 49, 43, 5, 4, 11, 44, 47, 35]:
        e = [0, 0, 0, 0, 1, 2, 1, 1, 1, 1, 1, 1]
        if e[chord1 % 12] == e[chord2 % 12]:
            return 0.5
        else:
            return 1
    elif d in [1, 2, 10]:
        return 2
    elif d in [96, 36, 48, 108, 114, 102]:
        return 3
    elif d in [12, 132]:
        return 4
    else:
        set1 = set(chord_dict[number_to_chord[chord1]])
        set2 = set(chord_dict[number_to_chord[chord2]])

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        jaccard_distance = 1 - intersection / union
        return 4 + jaccard_distance
#################Kemmeny clustering

# Define input parameters
n =  # Number of agents
k =  # Number of chords
UB =  # Upper bound for the number of parts


# Create a Gurobi model
model = gp.Model("Kemeny_Clustering")

# Create decision variables
a = {}  # Section associated with agent i
for i in range(n):
    for z in range(UB):
        a[i, z] = model.addVar(vtype=GRB.BINARY, name=f"a_{i}_{z}")
W = {}  # Chord selection variables
for j in range(k):
    W[j] = model.addVar(vtype=GRB.BINARY, name=f"W_{j}")

# Set objective function
model.setObjective(
    gp.quicksum(
        gp.quicksum(
            gp.quicksum(
                P[j, z, Z] * Q[i, z, Z] * chord_distance(b[i][j], W[j])
                for j in range(k)
            )
            for i in range(n)
        )
        for z in range(UB)
    ),
    sense=GRB.MINIMIZE,
)

# Add constraints
# Constraint 1
for i in range(n):
    for z in range(UB):
        model.addConstr(a[i, z] >= 0)
        model.addConstr(a[i, z] <= 1)
        model.addConstr(a[i, z] <= k)

# Constraint 2
for j in range(k):
    for z in range(UB - 1):
        model.addConstr(
            gp.quicksum(
                P[j, z, Z] * (Z[z + 1] - Z[z])
                for Z in range(UB)
            ) == 1
        )

# Constraint 3
for j in range(k):
    model.addConstr(
        gp.quicksum(
            P[j, UB - 1, Z] * (k - Z[UB - 1])
            for Z in range(UB)
        ) == 1
    )

# Add other constraints as needed

# Optimize the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found")
    # Extract and print the values of a[i, z] and W[j]
    # You can use model.getAttr("x", a[i, z]) and model.getAttr("x", W[j])
    # to retrieve the values of decision variables
    # Print the partition and chord selections based on the solution
else:
    print("No optimal solution found")

# Close the Gurobi model
model.close()

#################proportional algorithm
def distance_seq(seq_1, seq_2):
    # Assuming chord1 and chord2 are NumPy arrays of size k- It is supposed to be any the distance
    flag_arr=[]
    for i in range(len(seq_1)):
        flag.append(chord_distance(seq_1[i],seq_2[i]))
    return flag
        
        
#B[i] is the row of agent i
def proportional_algorithm(B, k):
    n = len(B)
    W =# Initialize a k sizied array of sequnce of chords W
    
    def objective_function(W):
        total = 0
        for i in range(n):
            sorted_distances = sorted(distance_seq(B[i], W), reverse=True)
            for j in range(k):
                total += (1 / (j + 1)) * (1 - sorted_distances[j])
        return total

    # Use a numerical optimization library to maximize the objective function

    

