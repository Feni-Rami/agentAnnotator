import ray  # type: ignore
import numpy as np
from collections import defaultdict
from ray.rllib.agents import ppo  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from fraud_detection_env import FraudDetectionEnvWithEconomics
from config import config
from malicious_annotator import (
    MaliciousAnnotator,
    AlwaysIncorrectAnnotator,
    PatternBasedMaliciousAnnotator,
    RandomMaliciousAnnotator,
    RandomStrategicMaliciousAnnotator,
)

# Ensure to import or define the EconomicSystem if needed in the main script

def get_suspended_annotators(annotators):
    return [i for i, annotator in enumerate(annotators) if annotator.suspended]

def setup_trainer_config(config):
    return {
        "env_config": config,
        "framework": "torch",
        "vf_clip_param": 60.0,
        "num_workers": 8,
        "ignore_worker_failures": True,
        "recreate_failed_workers": True,
    }

def replace_with_malicious_annotators(env, num_malicious_annotators):
    malicious_indices = np.random.choice(
        range(config["num_annotators"]), num_malicious_annotators, replace=False
    )
    for idx in malicious_indices:
        env.annotators[idx] = RandomMaliciousAnnotator(idx, config)
    return malicious_indices

def plot_and_save_results(results, costs, accuracy_history):
    plt.figure(figsize=(16, 10))
    plt.plot(results)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.title("Average Reward per Episode")
    plt.savefig("images/average_reward_per_episode.png")
    plt.close()

    plt.figure(figsize=(16, 10))
    plt.plot(costs, label="Total Cost per Episode", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Total Cost")
    plt.title("Cost per Episode Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("images/cost_per_episode.png")
    plt.close()

    plt.figure(figsize=(16, 10))
    for idx, accuracies in accuracy_history.items():
        plt.plot(accuracies, label=f"Annotator {idx}")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.title("Annotator Accuracy per Episode")
    plt.legend()
    plt.grid(True)
    plt.savefig("images/annotator_accuracy_per_episode.png")
    plt.close()

def analyze_final_accuracy(accuracy_history, malicious_indices):
    final_accuracies = {}
    for idx, accuracies in accuracy_history.items():
        if idx not in malicious_indices:
            final_accuracies[idx] = accuracies[-1]  # Final accuracy of the last episode

    avg_final_accuracy = np.mean(list(final_accuracies.values()))
    print(f"Average final accuracy of regular annotators: {avg_final_accuracy:.2f}")

    if avg_final_accuracy > 0.8:  # For example, an 80% accuracy threshold
        print("Annotators could provide correct annotations despite malicious ones.")
    else:
        print("The presence of malicious annotators compromised annotation accuracy.")

    # Plot the final accuracy of each annotator
    plt.figure(figsize=(16, 10))
    annotator_ids = list(final_accuracies.keys())
    accuracies = list(final_accuracies.values())
    plt.bar(annotator_ids, accuracies, color="blue")
    plt.axhline(0.8, color="red", linestyle="--", label="Accuracy Threshold (80%)")
    plt.xlabel("Annotator ID")
    plt.ylabel("Final Accuracy")
    plt.title("Final Accuracy of Each Non-Malicious Annotator")
    plt.legend()
    plt.grid(True)
    plt.savefig("images/final_accuracy_per_annotator.png")
    plt.close()

    return final_accuracies

def main():
    ray.init(ignore_reinit_error=True)
    env = FraudDetectionEnvWithEconomics(config)  # Use the updated environment
    trainer_config = setup_trainer_config(config)
    trainer = ppo.PPOTrainer(env=FraudDetectionEnvWithEconomics, config=trainer_config)

    # Replace some annotators with malicious annotators for the test
    num_malicious_annotators = int(config["num_annotators"] * 0.35)
    malicious_indices = replace_with_malicious_annotators(env, num_malicious_annotators)

    # Training simulation
    results = []
    accuracy_history = defaultdict(list)
    costs = []
    iteration = 10

    for episode in range(iteration):
        result = trainer.train()
        results.append(result["episode_reward_mean"])

        env.reset()
        current_episode_cost = 0

        for idx, _ in enumerate(config["data"]):
            observation = env._get_observation()
            action = trainer.compute_single_action(observation)  # Select an action
            observation, reward, done, info = env.step(action)

            if done:
                break

            current_episode_cost += info["cost"]

        env.performance_for_each_episode()
        costs.append(current_episode_cost)

        for idx, annotator in enumerate(env.annotators):
            accuracy_history[idx].append(annotator.accuracy)

    ray.shutdown()

    # Save results and individual annotator reputations in separate graphs
    plot_and_save_results(results, costs, accuracy_history)

    # Analyze the final accuracy of non-malicious annotators
    final_accuracies = analyze_final_accuracy(accuracy_history, malicious_indices)

    # Detect and evaluate the impact of malicious annotators
    malicious_annotators = env.server.detect_and_plot_malicious_annotators()
    env.server.evaluate_impact_with_and_without_malicious_annotators()
    env.perfomance()

    print(f"Chosen malicious annotators: {malicious_indices}")
    print(f"Detected malicious annotators after all episodes: {malicious_annotators}")
    print("Simulation complete. Reports and graphs have been saved.")

if __name__ == "__main__":
    main()
