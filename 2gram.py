import json

def create_2_gram_probability(chords_sequences):
    chords = [chord for sublist in chords_sequences for chord in sublist]

    chord_counts = {}
    chord_followers = {}

    for i in range(len(chords) - 1):
        current_chord = chords[i]
        next_chord = chords[i + 1]

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
            chord_probabilities[chord][follower] = follower_count / count

    return chord_probabilities

# Load the songs data from the JSON file
with open("songs.json", "r") as file:
    songs_data = json.load(file)

# Concatenate the chord sequences of all songs
all_chords_sequences = [song_data["chords"] for song_data in songs_data]

# Calculate the 2-gram probabilities for all chords
chord_probabilities_all_songs = create_2_gram_probability(all_chords_sequences)

# Example output for the final 2-gram model
for chord, followers in chord_probabilities_all_songs.items():
    print(f"Chord: {chord}")
    for follower, probability in followers.items():
        print(f"  -> {follower}: Probability: {probability:.3f}")
