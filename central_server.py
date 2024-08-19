from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

# Task complexity based on categories
category_complexity = {
    'Hypertension': 0.6,
    'Diabetes': 0.8,
    'Heart Disease': 0.7,
    'Kidney Disease': 0.7,
    'Liver Disease': 0.8
}

class CentralServer:
    def __init__(self, num_annotators: int):
        self.num_annotators = num_annotators
        self.reputation = {i: 1.0 for i in range(num_annotators)}
        self.time_taken_history = []
        self.current_annotation_time = None
        self.total_annotations = {i: 0 for i in range(num_annotators)}
        self.accuracy = {i: 0.0 for i in range(num_annotators)}
        self.features = {i: [] for i in range(num_annotators)}
        self.performance_history = defaultdict(list)
        self.malicious_detection_history = []
        self.episode_performance_history = defaultdict(list)

        # Economic variables
        self.tax_rate = 0.2  # Tax rate applied to annotators
        self.income_distribution = {i: 0.0 for i in range(num_annotators)}
        self.total_income_collected = 0.0

    def get_task_complexity(self, category):
        base_complexity = category_complexity.get(category, 0.5)
        # Dynamic complexity adjustment based on historical performance
        difficulty_factor = self._calculate_difficulty_factor()
        return base_complexity * difficulty_factor

    def _calculate_difficulty_factor(self) -> float:
        avg_accuracy = np.mean(list(self.accuracy.values()))
        accuracy_variance = np.var(list(self.accuracy.values()))        
        recent_improvement = 0
        for annotator_id in range(self.num_annotators):
            if len(self.performance_history[annotator_id]) > 1:
                recent_improvement += self.performance_history[annotator_id][-1] - self.performance_history[annotator_id][-2]
        recent_improvement /= self.num_annotators
        
        # Combine these factors into a more nuanced difficulty factor
        # Increase difficulty if average accuracy is high, variance is low (consistency), and recent improvement is positive
        difficulty_factor = 1.0 + (0.5 - avg_accuracy) - (0.2 * accuracy_variance) + (0.1 * recent_improvement)
        
        # Ensure the factor stays within reasonable bounds
        return max(0.5, min(difficulty_factor, 1.5))

    def apply_taxes(self, annotators):
        total_tax_collected = 0.0
        for annotator in annotators:
            annotator.apply_tax(self.tax_rate)
            total_tax_collected += annotator.tax
        self.total_income_collected = total_tax_collected

    def redistribute_income(self, annotators):
        equal_share = self.total_income_collected / len(annotators)
        for annotator in annotators:
            annotator.income += equal_share
            self.income_distribution[annotator.annotator_id] += equal_share

    def evaluate_annotations(self, annotations: List[str], truth: str) -> Tuple[List[float], List[float]]:
        accuracy_rewards = []
        previous_performance = []

        for annotator_id, annotator_annotations in enumerate(annotations):
            if not annotator_annotations:
                continue

            correct = annotator_annotations == truth
            complexity_weight = self.get_task_complexity(annotator_annotations)
            
            # Calculate performance based on complexity and correctness
            performance = (1.0 * complexity_weight) if correct else (-0.5 / complexity_weight)
            
            # Store performance history
            self.performance_history[annotator_id].append(performance)

            # Calculate previous performance excluding the current one
            if len(self.performance_history[annotator_id]) > 1:
                prev_perf = np.mean(self.performance_history[annotator_id][:-1])
            else:
                prev_perf = 0.0
            
            previous_performance.append(prev_perf)

            # Update total annotations and accuracy
            self.total_annotations[annotator_id] += 1
            correct_annotations = sum(1 for i in self.performance_history[annotator_id] if i > 0)
            accuracy = correct_annotations / self.total_annotations[annotator_id]
            accuracy_rewards.append(accuracy)

            # Update reputation based on current task performance
            self.reputation[annotator_id] += (0.1 * complexity_weight) if correct else (-0.1 * complexity_weight)
            
            # Store features: reputation and accuracy
            self.features[annotator_id].append([self.reputation[annotator_id], self.accuracy[annotator_id]])
            self.accuracy[annotator_id] = accuracy

        return accuracy_rewards, previous_performance

    def end_episode(self):
        for annotator_id in range(self.num_annotators):
            if self.performance_history[annotator_id]:
                average_performance = np.mean(self.performance_history[annotator_id])
                self.episode_performance_history[annotator_id].append(average_performance)
                self.performance_history[annotator_id] = []

    def plot_episode_performance(self):
        plt.figure(figsize=(16, 10))
        for annotator_id, performance in self.episode_performance_history.items():
            plt.plot(performance, label=f'Annotator {annotator_id}')

        plt.xlabel('Episodes')
        plt.ylabel('Average Performance')
        plt.title('Annotator Performance Per Episode')
        plt.legend()
        plt.grid(True)
        plt.savefig("images/annotator_episode_performance.png")
    
    def detect_and_plot_malicious_annotators(self) -> List[int]:
        feature_vectors, annotator_ids = self._prepare_feature_vectors()
        if len(feature_vectors) < 2:
            print("Not enough data for clustering.")
            return []

        feature_vectors = self._standardize_features(feature_vectors)
        clusters = self._cluster_annotators(feature_vectors)
        malicious_cluster = self._identify_malicious_cluster(feature_vectors, clusters)
        malicious_annotators = self._get_malicious_annotators(clusters, malicious_cluster, annotator_ids)
        
        self._plot_clusters(feature_vectors, clusters, malicious_cluster, annotator_ids)

        return malicious_annotators

    def _cluster_annotators(self, feature_vectors: np.ndarray) -> np.ndarray:
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=42)
        return kmeans.fit_predict(feature_vectors)

    def _prepare_feature_vectors(self) -> Tuple[np.ndarray, List[int]]:
        feature_vectors, annotator_ids = [], []
        for annotator_id in range(self.num_annotators):
            if self.features[annotator_id]:
                feature_vectors.append(self.features[annotator_id][-1])
                annotator_ids.append(annotator_id)
        return np.array(feature_vectors), annotator_ids

    def _standardize_features(self, feature_vectors: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        return scaler.fit_transform(feature_vectors)

    def _identify_malicious_cluster(self, feature_vectors: np.ndarray, clusters: np.ndarray) -> int:
        cluster_0 = feature_vectors[clusters == 0]
        cluster_1 = feature_vectors[clusters == 1]

        avg_rep_0, avg_acc_0 = np.mean(cluster_0, axis=0)[:2]
        avg_rep_1, avg_acc_1 = np.mean(cluster_1, axis=0)[:2]

        return 0 if (avg_rep_0 + avg_acc_0) < (avg_rep_1 + avg_acc_1) else 1

    def _get_malicious_annotators(self, clusters: np.ndarray, malicious_cluster: int, annotator_ids: List[int]) -> List[int]:
        return [annotator_ids[i] for i in range(len(clusters)) if clusters[i] == malicious_cluster]

    def _plot_clusters(self, feature_vectors: np.ndarray, clusters: np.ndarray, malicious_cluster: int, annotator_ids: List[int]):
        # Color settings
        colors = ['red' if cluster == malicious_cluster else 'blue' for cluster in clusters]
        plt.figure(figsize=(16, 10))
        
        # Scatter plot with enhanced visibility
        scatter = plt.scatter(
            feature_vectors[:, 0], 
            feature_vectors[:, 1], 
            c=colors, 
            alpha=0.6, 
            edgecolors='k',  # Use black edges for better contrast
            s=200  # Larger markers for better visibility
        )

        # Annotating each point with the annotator ID
        for i in range(len(feature_vectors)):
            plt.annotate(
                annotator_ids[i], 
                (feature_vectors[i, 0], feature_vectors[i, 1]), 
                fontsize=12, 
                ha='right',
                color='black'  # Use black color for better readability
            )

        plt.xlabel("Reputation (Standardized)")
        plt.ylabel("Accuracy (Standardized)")
        plt.title("Annotator Clusters with Malicious Detection")
        plt.grid(True)
        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Normal'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Malicious')
        ])
        plt.savefig("images/annotator_clusters.png")
        plt.close()

    def evaluate_impact_with_and_without_malicious_annotators(self):
        malicious_annotators = self.detect_and_plot_malicious_annotators()

        all_reputations = list(self.reputation.values())
        all_accuracies = list(self.accuracy.values())

        non_malicious_reputations = [self.reputation[i] for i in range(self.num_annotators) if i not in malicious_annotators]
        non_malicious_accuracies = [self.accuracy[i] for i in range(self.num_annotators) if i not in malicious_annotators]

        self._plot_impact_comparison(all_reputations, non_malicious_reputations, "Reputation")
        self._plot_impact_comparison(all_accuracies, non_malicious_accuracies, "Accuracy")

    def _plot_impact_comparison(self, all_values, non_malicious_values, metric_name):
        plt.figure(figsize=(16, 10))
        sns.kdeplot(all_values, label='All Annotators', shade=True)
        sns.kdeplot(non_malicious_values, label='Non-Malicious Annotators', shade=True)
        plt.title(f'Impact on {metric_name} with and without Malicious Annotators')
        plt.xlabel(f'{metric_name}')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"images/impact_comparison_{metric_name}.png")
