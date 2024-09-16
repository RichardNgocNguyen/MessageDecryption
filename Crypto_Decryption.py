from collections import Counter
import numpy as np
import random


# ---------------------------------------------------------------------------------------------------------------------
def caesar_cipher(crypto):
    # Shifting letters in the message
    alphabet_dict = {"A": "B", "K": "L", "U": "V",
                     "B": "C", "L": "M", "V": "W",
                     "C": "D", "M": "N", "W": "X",
                     "D": "E", "N": "O", "X": "Y",
                     "E": "F", "O": "P", "Y": "Z",
                     "F": "G", "P": "Q", "Z": "A",
                     "G": "H", "Q": "R",
                     "H": "I", "R": "S",
                     "I": "J", "S": "T",
                     "J": "K", "T": "U",

                     }
    # Try each rotation and store the result
    rotations = []
    message = crypto.replace(" ", "").upper()
    for alphabet in range(25):
        temp_message = ""
        for letter in message:
            temp_message += alphabet_dict[letter]
        message = temp_message
        rotations.append(message)

    # Borrows the text fitness function from the genetic algorithm to return result with best score
    fitness_record = []
    for text in rotations:
        fitness_record.append(fitness(text))

    s = sorted(zip(fitness_record, rotations))
    decrypted = [s2 for s1, s2 in s][0]
    return decrypted


# ---------------------------------------------------------------------------------------------------------------------
def generate_initial(m, quantity):
    # Counts the number of occurrences for each letter
    msg = m.replace(" ", "").upper()

    freq = {
        'A': 8.55, 'K': 0.81, 'U': 2.68,
        'B': 1.60, 'L': 4.21, 'V': 1.06,
        'C': 3.16, 'M': 2.53, 'W': 1.83,
        'D': 3.87, 'N': 7.17, 'X': 0.19,
        'E': 12.1, 'O': 7.47, 'Y': 1.72,
        'F': 2.18, 'P': 2.07, 'Z': 0.11,
        'G': 2.09, 'Q': 0.10,
        'H': 4.96, 'R': 6.33,
        'I': 7.33, 'S': 6.73,
        'J': 0.22, 'T': 8.94,
    }

    # Variable s stores a tuple of (char probability, char) sorted in order of likelihood
    char = freq.keys()
    probabilities = [freq[c] for c in char]
    s = sorted(zip(probabilities, char), reverse=True)

    # Stores letter in the order of frequency and inserts missing characters at any random point
    msg_freq = [m for m, val in Counter(msg).most_common()]
    for c in char:
        if c not in msg_freq:
            random_index = np.random.randint(0, len(msg_freq))
            msg_freq.insert(random_index, c)

    # Stores key combinations
    p = set()
    for i in range(quantity):
        char = [s2 for s1, s2 in s]
        probabilities = [s1 for s1, s2 in s]
        # Creates a key mapping based on general letter freq
        mapping = ""
        for j in range(len(alphabet)):
            # Given weighted choices, randomly pick a letter and add to mapping
            c = random.choices(population=char, weights=probabilities, k=1)[0]
            mapping += c
            # Letter weight is set to 0 to avoid repeats
            probabilities[char.index(c)] = 0
        # Maps message letter freq to general letter frequency
        encoding = []
        for m in range(len(msg_freq)):
            encoding.append([msg_freq[m], mapping[m]])
        # Sorts the encoding (plaintext, key) alphabetically to get the corresponding decryption key
        # Add it to the population
        key = ""
        for pair in sorted(encoding):
            key += pair[1]

        p.add(key)
    return list(p)


def selection(p, w):
    # Picks two different parents
    populace, weights = p, w
    parents = random.choices(population=populace, weights=weights, k=2)
    while len(set(parents)) == 1:
        parents = random.choices(population=populace, weights=weights, k=2)

    return parents


def crossover_point(keyA, keyB):
    # Implant material from other parent from random point
    point = random.randint(1, 25)

    # Crossovers will introduce duplicates typically in one to one mappings
    A1 = keyA[:point]
    A2 = keyB[point:]  # Material from B
    A = list(A1 + A2)

    B1 = keyB[:point]
    B2 = keyA[point:]  # Material from A
    B = list(B1 + B2)

    # Resolve duplicates as a result of crossovers for A
    remaining = set(alphabet)
    if len(A) != len(set(A)):
        duplicates = []
        for letter in A1:
            remaining.remove(letter)
        for i, letter in enumerate(A2):
            if letter not in remaining:
                duplicates.append(point + i)
            else:
                remaining.remove(letter)
        for i, unused in enumerate(remaining):
            A[duplicates[i]] = unused

    # Resolve duplicates as a result of crossovers for B
    remaining = set(alphabet)
    if len(B) != set(B):
        duplicates = []
        for letter in B1:
            remaining.remove(letter)
        for i, letter in enumerate(B2):
            if letter not in remaining:
                duplicates.append(point + i)
            else:
                remaining.remove(letter)
        for i, unused in enumerate(remaining):
            B[duplicates[i]] = unused

    return ''.join(A), ''.join(B)


def mutation(key, probability=0.5):
    # Rolls on whether to mutate
    if np.random.randint(0, 1000) / 1000 >= probability:
        # Makes sure A and B are not the same
        A = np.random.randint(0, 26)
        B = np.random.randint(0, 26)
        while A == B:
            B = np.random.randint(0, 26)
        # Create new key from mutation
        new_key = ""
        for index, l in enumerate(key):
            if index == A:
                new_key += key[B]
            elif index == B:
                new_key += key[A]
            else:
                new_key += l
        return new_key
    return key


def fitness(text):
    # We will calculate fitness of a text
    n = 4
    fit = 0
    msg = text.replace(" ", "").upper()
    for i in range(len(msg) - n + 1):
        gram = msg[i:i + n]
        if gram in record:
            fit += record[gram]
        else:
            fit += O

    avg = fit / (len(msg) - n + 1)
    fit = abs(avg - f_normal) / f_normal

    return fit


def genetic_algorithm(message, generation):
    # Initial generation is created and each individual is assigned a fitness score
    population_size = 200
    fitness_record = []
    population = generate_initial(m=message, quantity=population_size * 10)
    for individual in population:
        mapping = {}
        for i in range(len(alphabet)):
            mapping[alphabet[i]] = individual[i]
        decrypted = ""
        for letter in message.upper():
            if letter != " ":
                decrypted += mapping[letter]
        score = fitness(decrypted)
        fitness_record.append(score)

    # Iterates through each generation
    print("Generation", end=" ")
    for g in range(generation):
        if g % 100 == 0:
            print(g, end=" ")

        # Sorts population in the order from best to worst fitness
        s = sorted(zip(fitness_record, population))
        population = [s2 for s1, s2 in s]

        # Save the best 100 individuals for the next generation
        next_generation = set()
        for individual in population:
            next_generation.add(individual)
            if len(next_generation) == 20:
                break

        # While population size is not met...
        # We will crossover two parents to create two children
        # Randomly decide to mutate
        while len(next_generation) < population_size:
            parents = selection(population, fitness_record)
            childA, childB = crossover_point(''.join(parents[0]), ''.join(parents[1]))
            childA = mutation(childA)
            childB = mutation(childB)

            next_generation.add(childA)
            next_generation.add(childB)

        # Calculate the fitness of population for the next generation
        population = list(next_generation)
        fitness_record = []
        for individual in population:
            mapping = {}
            for i in range(len(alphabet)):
                mapping[alphabet[i]] = individual[i]
            decrypted = ""
            for letter in message.upper():
                if letter != " ":
                    decrypted += mapping[letter]
            score = fitness(decrypted)
            fitness_record.append(score)

    # Sort final population based on fitness and returns the result
    s = sorted(zip(fitness_record, population))
    population = [s2 for s1, s2 in s]

    print("\n")
    return population


# ---------------------------------------------------------------------------------------------------------------------
def main():
    # Run a evolution instance per encrypted message
    for i, e_msg in enumerate(encrypted_msg):
        print("Decrypting Message:", e_msg)
        # Run caesar cipher for the first and second crypto text
        if i < 2:
            text = caesar_cipher(e_msg)
            print("Message:", text, "\n")

        # Run a genetic algorithm for the third and fourth crypto text
        else:
            generations = 2100
            population = genetic_algorithm(e_msg, generations)
            # List top 3 results
            population = population[0:3]
            for individual in population:
                mapping = {}
                for i in range(len(alphabet)):
                    mapping[alphabet[i]] = individual[i]
                decrypted = ""
                for letter in e_msg.upper():
                    if letter != " ":
                        decrypted += mapping[letter]
                print("Message:", decrypted)
            # May take a few tries to get the right message
            while True:
                x = input("Wrong Message? Try Again(Y/N): ")
                print()
                if x.upper() == "Y":
                    population = genetic_algorithm(e_msg, generations)
                    population = population[0:3]
                    for individual in population:
                        mapping = {}
                        for i in range(len(alphabet)):
                            mapping[alphabet[i]] = individual[i]
                        decrypted = ""
                        for letter in e_msg.upper():
                            if letter != " ":
                                decrypted += mapping[letter]
                        print("Message:", decrypted)
                else:
                    break


# ---------------------------------------------------------------------------------------------------------------------
# Extract Info from Quad-gram Txt
total = 0
record = {}
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
n_gram = open("txt/Quadgrams.txt")

# Match quadgram key to its occurrence
# Get total sample size
# Store Top 20,000 occurring quadgrams
top_occurring = []
for i, line in enumerate(n_gram):
    gram, occur = line.split(" ")
    if i < 20000:
        top_occurring.append(gram)
    record[gram] = int(occur)
    total += int(occur)

# Calculate log probabilities and match it to its respective key
for key in record:
    record[key] = np.log10(float(record[key] / total))
O = np.log10(0.01 / total)

# Calculate fitness normalized for the text fitness function
f_normal = 0
for quadgram in top_occurring:
    f_normal += record[quadgram]
f_normal = -f_normal / len(top_occurring)

# Encrypted messages
encrypted_msg = []
for line in open("txt/crypto_code.txt"):
    encrypted_msg.append(line[2:-1])

main()
