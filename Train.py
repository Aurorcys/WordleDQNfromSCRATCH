import numpy as np 
import pandas as pd 


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


# ==========
# Training Loop Time

import random


# Load both
with open('valid_guesses.csv', 'r') as f:
    valid_words = [line.strip() for line in f.readlines() if len(line.strip()) == 5 and line.strip().isalpha()]

with open('valid_solutions.csv', 'r') as f:
    answers = [line.strip() for line in f.readlines() if len(line.strip()) == 5 and line.strip().isalpha()]

# Combine — all words in one vocabulary
all_words = list(set(valid_words + answers))
word_to_idx = {word: i for i, word in enumerate(all_words)}


from NeuralNetwork import NN

model = NN(all_words)

epsilon = 1.0
episodes = 100000
gamma = 0.9


for episode in range(episodes):
    answer = random.choice(answers)
    possible = answers.copy()
    state = encode_state(possible)

    episode_steps = []

    for turn in range(6):
        if random.random() < epsilon:
            guess = random.choice(possible) if possible else random.choice(valid_words)
        
        else:
            q_values = model.forward(state, possible)
            best_idx = np.argmax(q_values)
            guess = all_words[best_idx]

        
        feedback = get_feedback(guess, answer)
        done = (feedback == (2, 2, 2, 2, 2))
        reward = 10 if done else -1

        possible = filter_answers(possible, guess, feedback)
        next_state = encode_state(possible)

        episode_steps.append((state, guess, reward, next_state, done))
        state = next_state

        if done:
            break
    
    #update time
    for state, guess, reward, next_state, done in reversed(episode_steps):
        if done:
            target = reward
        else:

            _ = model.forward(next_state, possible)
            target = reward + gamma * np.max(_)
        

        model.forward(state, possible)
        model.backward(target, word_to_idx[guess])


    epsilon = max(0.1, epsilon * 0.9999)  

    if episode % 100 == 0:
        print(f"Episode {episode}, epsilon={epsilon:.3f}, last game guesses={len(episode_steps)}")

print("Training complete!")


    
np.savez('wordle_dqn.npz',
         W1=model.W1, b1=model.b1,
         W2=model.W2, b2=model.b2,
         W3=model.W3, b3=model.b3)


