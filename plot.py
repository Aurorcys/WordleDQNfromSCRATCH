import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import random

import matplotlib.pyplot as plt
import random



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


with open('data/valid_guesses.csv', 'r') as f:
    valid_words = [line.strip() for line in f.readlines() if len(line.strip()) == 5 and line.strip().isalpha()]

with open('data/valid_solutions.csv', 'r') as f:
    answers = [line.strip() for line in f.readlines() if len(line.strip()) == 5 and line.strip().isalpha()]

# Combine — all words in one vocabulary
all_words = list(set(valid_words + answers))

from NeuralNetwork import NN

model = NN(all_words)

data = np.load('data/wordle_dqn.npz')
W1, b1 = data['W1'], data['b1']
W2, b2 = data['W2'], data['b2']
W3, b3 = data['W3'], data['b3']

# Overwrite weights
model.W1, model.b1 = W1, b1
model.W2, model.b2 = W2, b2
model.W3, model.b3 = W3, b3

def animate_multiple_games(model, n_games=120):
    total_guesses = 0
    wins = 0
    guess_history = []
    
    plt.ion()
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('#0d0d1a')
    
    # Grid: board | stats | histogram
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.8, 1.2])
    ax_board = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])
    ax_hist = fig.add_subplot(gs[2])
    
    for game_num in range(1, n_games + 1):
        answer = random.choice(answers)
        possible = answers.copy()
        state = encode_state(possible)
        guesses = []
        feedbacks = []
        
        for turn in range(6):
            q_values = model.forward(state, possible)
            guess = all_words[np.argmax(q_values)]
            feedback = get_feedback(guess, answer)
            
            guesses.append(guess)
            feedbacks.append(feedback)
            
            # ---- BOARD ----
            ax_board.clear()
            ax_board.set_facecolor('#0d0d1a')
            ax_board.set_xlim(0, 5)
            ax_board.set_ylim(0, 6)
            ax_board.invert_yaxis()
            ax_board.axis('off')
            
            for row in range(6):
                for col in range(5):
                    if row < len(guesses):
                        fb = feedbacks[row][col]
                        color = '#00ff88' if fb == 2 else '#ffcc00' if fb == 1 else '#333344'
                        text = guesses[row][col].upper()
                        text_color = 'black' if fb in (1, 2) else 'white'
                    else:
                        color = '#1a1a2e'
                        text = ''
                        text_color = 'white'
                    
                    rect = plt.Rectangle((col, row), 1, 1, fill=True, facecolor=color,
                                        edgecolor='white', linewidth=2)
                    ax_board.add_patch(rect)
                    if text:
                        ax_board.text(col + 0.5, row + 0.5, text, fontsize=26, fontweight='bold',
                                     color=text_color, ha='center', va='center')
            
            ax_board.set_title(f'Game {game_num}/{n_games}', fontsize=14, color='white', fontweight='bold')
            
            # ---- STATS ----
            ax_stats.clear()
            ax_stats.set_facecolor('#0d0d1a')
            ax_stats.axis('off')
            avg = total_guesses / wins if wins else 0
            win_rate = wins / game_num * 100 if game_num > 0 else 0
            
            stats = f"""
GAME {game_num}/{n_games}

Wins: {wins}
Losses: {game_num - wins}
Win Rate: {win_rate:.1f}%
Avg Guesses: {avg:.2f}

Best: {min(guess_history) if guess_history else '—'}
Worst: {max(guess_history) if guess_history else '—'}

Answer: {answer.upper()}
            """
            ax_stats.text(0.1, 0.5, stats, fontsize=15, color='white', fontfamily='monospace',
                         verticalalignment='center')
            
            # ---- HISTOGRAM ----
            ax_hist.clear()
            ax_hist.set_facecolor('#0d0d1a')
            if guess_history:
                counts = [guess_history.count(i) for i in range(1, 8)]
                bars = ax_hist.bar(range(1, 8), counts, color=['#00ff88']*6 + ['#ff6b6b'],
                                  edgecolor='white', linewidth=1.5)
                ax_hist.set_xlabel('Guesses Needed', fontsize=12, color='white')
                ax_hist.set_ylabel('Count', fontsize=12, color='white')
                ax_hist.set_title('Guess Distribution', fontsize=14, color='white', fontweight='bold')
                ax_hist.set_xticks(range(1, 8))
                ax_hist.set_xticklabels(['1', '2', '3', '4', '5', '6', 'Fail'])
                ax_hist.tick_params(colors='white')
                ax_hist.set_ylim(0, max(counts) + 2)
                for bar, count in zip(bars, counts):
                    if count > 0:
                        ax_hist.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                                   str(count), ha='center', color='white', fontweight='bold')
            
            plt.draw()
            plt.pause(0.2)
            
            if feedback == (2, 2, 2, 2, 2):
                wins += 1
                total_guesses += turn + 1
                guess_history.append(turn + 1)
                break
            
            possible = filter_answers(possible, guess, feedback)
            state = encode_state(possible)
        
        # If failed
        if feedback != (2, 2, 2, 2, 2):
            guess_history.append(7)
        
        plt.pause(0.5)
    
    plt.ioff()
    plt.show()
    
    print(f"\nFinal: {wins}/{n_games} wins ({wins/n_games*100:.1f}%)")
    print(f"Average guesses: {total_guesses/wins:.2f}" if wins else "No wins")


animate_multiple_games(model)