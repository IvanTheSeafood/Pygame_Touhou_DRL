import matplotlib.pyplot as plt
from assets.scripts.learning import mlData

def plot_scores(episode_scores, episode_number):
    # Plot scores for the episode
    plt.figure(1)
    plt.plot(range(1, len(episode_scores) + 1), episode_scores, label=f'Episode {episode_number}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs Time Step (Episode {})'.format(episode_number))
    plt.grid(True)
    plt.legend()  # Display legend with automatically generated labels
    plt.pause(0.001)  # Pause briefly to update the plot
    plt.savefig(mlData.version+'Total Reward vs Time Step.png')

def plot_highest_scores(scores):
    plt.figure(2)  # Set the figure number to 2
    episodes = list(range(1, len(scores) + 1))  # Generate the list of episodes starting from 1
    plt.plot(episodes, scores, marker='x', linestyle='None', markersize=10)

    # Add a tag for each point indicating the episode number
    #for i, score in enumerate(scores, start=1):
        #plt.text(i, score, f'Episode {i}', ha='center', va='bottom')

    plt.plot(episodes, scores, marker='', linestyle='-')

    plt.title('Highest Score After Each Episode')
    plt.xlabel('Episode')
    plt.ylabel('Highest Score')
    plt.grid(True)
    plt.legend()  # Display legend with automatically generated labels
    plt.pause(0.001)  # Pause briefly to update the plot
    plt.savefig(mlData.version+'Highest Score After Each Episode.png')