import numpy as np 
import pandas as pd 


# =========
#Helper functions

def get_feedback(guess, answer):
    feedback = [0] * 5
    answer_chars = list(answer)

    for i in range(5):
        if guess[i] == answer[i]:
            feedback[i] = 2
            answer_chars[i] = None

    for i in range(5):
        if feedback[i] == 0 and guess[i] in answer_chars:
            feedback[i] = 1
            answer_chars[answer_chars.index(guess[i])] = None

    return tuple(feedback)


def filter_answers(possible, guess, feedback):
    new_possible = []

    for word in possible:
        if get_feedback(guess, word) == feedback:
            new_possible.append(word)
        
    return new_possible


def encode_state(possible):
    state = np.zeros(5 * 26)  # 130 zeros
    
    for word in possible:
        for pos, char in enumerate(word):
            idx = pos * 26 + (ord(char) - ord('a'))
            state[idx] += 1
    
    return state / len(possible)

# =========




data = np.load('data/wordle_dqn.npz')
W1, b1 = data['W1'], data['b1']
W2, b2 = data['W2'], data['b2']
W3, b3 = data['W3'], data['b3']

with open('data/valid_guesses.csv', 'r') as f:
    valid_words = [line.strip() for line in f.readlines() if len(line.strip()) == 5 and line.strip().isalpha()]

with open('data/valid_solutions.csv', 'r') as f:
    answers = [line.strip() for line in f.readlines() if len(line.strip()) == 5 and line.strip().isalpha()]

# Combine — all words in one vocabulary
all_words = list(set(valid_words + answers))

# Rebuild model
from NeuralNetwork import NN
model = NN(all_words)  

# Overwrite weights
model.W1, model.b1 = W1, b1
model.W2, model.b2 = W2, b2
model.W3, model.b3 = W3, b3


def play_wordle(model, answer):
    possible = answers.copy()
    state = encode_state(possible)

    for turn in range(6):
        q_values = model.forward(state, possible)
        guess = all_words[np.argmax(q_values)]
        print(f"Turn {turn + 1}: {guess}")

        feedback = get_feedback(guess, answer)
        if feedback == (2, 2, 2, 2, 2):
            print(f"Solved on turn {turn + 1}! Answer: {answer}")
            return turn + 1
        
        possible = filter_answers(possible, guess, feedback)
        state = encode_state(possible)

    print(f"Failed! Answer was {answer}")
    return 7


play_wordle(model, "crane")


"""total_guesses = 0
wins = 0

for answer in answers:
    guesses = play_wordle(model, answer)
    if guesses <= 6:
        wins += 1
        total_guesses += guesses

print(f"Win rate: {wins}/{len(answers)} = {wins/len(answers)*100:.1f}%")
print(f"Average guesses (wins): {total_guesses/wins:.2f}")"""

"""
Win rate: 2240/2315 = 96.8%
Average guesses (wins): 4.20
"""



"""#using it myself

def play_wordle_myself(model):
    possible = answers.copy()
    state = encode_state(possible)

    for turn in range(6):
        q_values = model.forward(state, possible)
        guess = all_words[np.argmax(q_values)]
        print(f"Turn {turn + 1}: {guess}")
        
        while True:
            feedback_str = input("Type feedback (e.g., 2,0,1,0,0): ").strip()
            try:
                feedback = tuple(map(int, feedback_str.split(',')))
                if len(feedback) == 5 and all(f in (0, 1, 2) for f in feedback):
                    break
                print("Must be 5 numbers (0, 1, or 2) separated by commas.")
            except ValueError:
                print("Invalid format. Use numbers separated by commas, e.g., 2,0,1,0,0")

        if feedback == (2, 2, 2, 2, 2):
            print(f"Solved on turn {turn + 1}!")
            return turn + 1
        
        possible = filter_answers(possible, guess, feedback)
        state = encode_state(possible)

    print("Failed!")
    return 7

play_wordle_myself(model)"""













