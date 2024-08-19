import gym
from gym import spaces
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from annotator_agent import AnnotatorAgent
from central_server import CentralServer
from typing import Dict, Tuple, List
from collections import Counter

class EconomicSystem:
    def __init__(self, tax_rate=0.3, redistribution_weight=0.5):
        self.tax_rate = tax_rate
        self.redistribution_weight = redistribution_weight

    def apply_taxation(self, annotators):
        total_income = sum(annotator.income for annotator in annotators)
        taxed_amount = total_income * self.tax_rate
        redistributed_amount = taxed_amount * self.redistribution_weight / len(annotators)

        for annotator in annotators:
            annotator.income -= annotator.income * self.tax_rate
            annotator.income += redistributed_amount

class AnnotationRewardSystem:
    def __init__(self, reward_type: str):
        self.reward_type = reward_type

    def calculate_reward(self, annotations: List[str], ground_truth: str, action: str, complexity: float = 0.5, previous_performance: float = 0.0, community_interaction: int = 0) -> float:
        if self.reward_type == "quality":
            return self.quality_reward(annotations, ground_truth, complexity, action)
        elif self.reward_type == "consistency":
            return self.consistency_reward(annotations, action)
        elif self.reward_type == "improvement":
            return self.improvement_reward(previous_performance, annotations, ground_truth, action)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def quality_reward(self, annotations: List[str], ground_truth: str, complexity: float, action: str) -> float:
        reward = 0.0
        for annotation in annotations:
            if annotation == ground_truth:
                base_reward = 100.0 if complexity > 0.7 else 50.0
                reward += base_reward * (1.1 if action == 'confirmation' else 1.2 if action == 'proofreading' else 1.0)
        return reward

    def consistency_reward(self, annotations: List[str], action: str) -> float:
        unique_annotations = set(annotations)
        penalty = len(unique_annotations) * -5.0
        base_reward = 50.0 + penalty
        return max(0.0, base_reward * (1.1 if action == 'confirmation' else 1.2 if action == 'proofreading' else 1.0))

    def improvement_reward(self, previous_performance: float, annotations: List[str], ground_truth: str, action: str) -> float:
        current_performance = self.measure_current_performance(annotations, ground_truth)
        improvement = current_performance - previous_performance
        base_reward = max(0.0, improvement * 100.0)
        return base_reward * (1.1 if action == 'confirmation' else 1.2 if action == 'proofreading' else 1.0)

    def measure_current_performance(self, annotations: List[str], ground_truth: str) -> float:
        correct_annotations = sum(1 for annotation in annotations if annotation == ground_truth)
        total_annotations = len(annotations)
        return correct_annotations / total_annotations if total_annotations > 0 else 0.0

    def calculate_community_interaction(self, annotations: List[str]) -> float:
        if len(annotations) <= 1:
            return 0.0  
        
        diversity_score = self._calculate_diversity(annotations)        
        similarity_score = self._calculate_similarity(annotations)        
        frequency_score = self._calculate_frequency_score(annotations)        
        interaction_score = (diversity_score * 0.4) + (similarity_score * 0.3) + (frequency_score * 0.3)
        
        return interaction_score

    def _calculate_diversity(self, annotations: List[str]) -> float:
        unique_annotations = set(annotations)
        diversity = len(unique_annotations) / len(annotations)
        return diversity

    def _calculate_similarity(self, annotations: List[str]) -> float:
        unique_annotations = list(set(annotations))
        if len(unique_annotations) <= 1:
            return 0.0 

        similarities = []
        for i in range(len(unique_annotations)):
            for j in range(i + 1, len(unique_annotations)):
                set1 = set(unique_annotations[i])
                set2 = set(unique_annotations[j])
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                sim = intersection / union if union != 0 else 0
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity  # Invert similarity for interaction score


    def _calculate_frequency_score(self, annotations: List[str]) -> float:
        counter = Counter(annotations)
        max_freq = max(counter.values())
        frequency_score = 1.0 - (max_freq / len(annotations))  # Higher spread gives higher score
        return frequency_score

    def _string_to_vector(self, s: str) -> np.ndarray:
        return np.array([ord(c) for c in s])

class FraudDetectionEnv(gym.Env):
    def __init__(self, config: Dict):
        super(FraudDetectionEnv, self).__init__()
        
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(config["data"])

        self.time_budget = np.random.uniform(*config.get("time_budget_range", (0.8, 1.2)))
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.data.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(config["num_actions"])
        self.action_mapping = ['confirmation', 'proofreading']
        self.annotators = [AnnotatorAgent(id, config) for id in range(config["num_annotators"])]
        self.server = CentralServer(config["num_annotators"])
        self.truth = config["truth"]
        self.reputation = {annotator.annotator_id: 1.0 for annotator in self.annotators}
        self.current_index = 0
        self.time_taken_history = []
        self.reward_system = AnnotationRewardSystem(reward_type="quality") 
        self.tax_rate = config.get("tax_rate", 0.1)  # New: Tax rate for income redistribution
        self.income_distribution = []

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        done = self.current_index >= len(self.data)
        if done:
            return self._get_observation(), 0, done, {}

        observation = self._get_observation()
        annotations, states, actions = self._collect_annotations(observation)
        true_label = self.truth[self.current_index]

        accuracy_rwd, previous_performance = self.server.evaluate_annotations(annotations, true_label)
        previous_perf = previous_performance[0] if previous_performance else 0.0
        reward = self.reward_system.calculate_reward(
            annotations,
            true_label,
            action=self.action_mapping[action],
            complexity=self._calculate_data_complexity(observation),
            previous_performance=previous_perf,
            community_interaction=self.reward_system.calculate_community_interaction(annotations)
        )

        accuracy_reward = sum(accuracy_rwd) / len(accuracy_rwd)
        total_reward = reward + accuracy_reward  # Combine the action reward and the accuracy reward
        self.current_index += 1 

        cost = self.calculate_cost()
        adjusted_reward = total_reward - cost

        self._apply_taxation()  # New: Apply taxation to redistribute income

        next_observation = self._get_observation()
        next_states = [annotator._get_state(self._calculate_data_complexity(next_observation), 
                                            self._allocate_time(self._calculate_data_complexity(next_observation))) for annotator in self.annotators]

        self._provide_feedback_to_annotators(annotations, states, actions, true_label, next_states)
        done = self.current_index >= len(self.data)
        return self._get_observation(), adjusted_reward, done, {"cost": cost}

    def _apply_taxation(self):
        # New: Collect income from annotators, apply tax, and redistribute
        total_income = sum(annotator.income for annotator in self.annotators)
        taxed_amount = total_income * self.tax_rate
        redistributed_amount = taxed_amount / len(self.annotators)

        for annotator in self.annotators:
            annotator.income -= annotator.income * self.tax_rate
            annotator.income += redistributed_amount

        self.income_distribution.append([annotator.income for annotator in self.annotators])

    def reset(self) -> np.ndarray:
        self.current_index = 0
        self.reputation = {annotator.annotator_id: 1.0 for annotator in self.annotators}
        self.server.time_taken_history = []
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        if self.current_index < len(self.data):
            return self.data[self.current_index].astype(np.float32)
        else:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

    def performance_for_each_episode(self):
        self.server.end_episode()

    def perfomance(self):
        self.server.plot_episode_performance()

    def calculate_cost(self) -> float:
        alpha = 1.0
        beta = 1.0
        gamma = 1.0
        max_time = max(self.server.time_taken_history) if self.server.time_taken_history else 1.0

        cost = 0
        for annotator in self.annotators:
            data_complexity = annotator._calculate_data_complexity(self._get_observation())
            annotation_time = annotator._allocate_time(data_complexity)
            
            if self.current_index < len(self.truth):
                deviation_from_truth = 1 if annotator.annotation != self.truth[self.current_index] else 0
            else:
                deviation_from_truth = 1 
                
            accuracy = annotator.reputation

            normalized_accuracy = max(0, min(accuracy, 1))
            normalized_time = annotation_time / max_time

            cost += (alpha * deviation_from_truth +
                     beta * (1 - normalized_accuracy) +
                     gamma * normalized_time)
        return cost

    def _calculate_data_complexity(self, observation: np.ndarray) -> float:
        if len(observation) is None:
            return 1.0

        normalized_observation = (observation - np.min(observation)) / (np.ptp(observation) + 1e-6)
        std_dev = np.std(normalized_observation)
        hist, _ = np.histogram(normalized_observation, bins=10, density=True)
        hist += 1e-6  # Add a small constant to avoid log(0)
        entropy = -np.sum(hist * np.log(hist))

        mean = np.mean(normalized_observation)
        complexity = 1.0 - (mean + std_dev + entropy) / 3.0
        complexity = max(0, min(complexity, 1))

        return complexity

    def _allocate_time(self, data_complexity: float) -> float:
        time_allocation = self.time_budget * (1 - data_complexity) * np.random.uniform(0.8, 1.2)
        min_time = 0.1
        max_time = self.time_budget * 1.5
        time_allocation = np.clip(time_allocation, min_time, max_time)
        return time_allocation

    def _collect_annotations(self, observation: np.ndarray) -> Tuple[List[str], List[Tuple], List[str]]:
        annotations = []
        states = []
        actions = []

        for annotator in self.annotators:
            annotation, state, annotator_action = annotator.annotate_data(observation)
            annotations.append(annotation)
            states.append(state)
            actions.append(annotator_action)

        return annotations, states, actions

    def _provide_feedback_to_annotators(self, annotations: List[str], states: List[Tuple], actions: List[str], true_label: str, next_states: List[Tuple]):
        for i, annotator in enumerate(self.annotators):
            correct = annotations[i] == true_label
            annotator.receive_feedback(correct, states[i], actions[i], next_states[i])

class FraudDetectionEnvWithEconomics(FraudDetectionEnv):
    def __init__(self, config):
        super(FraudDetectionEnvWithEconomics, self).__init__(config)
        self.economic_system = EconomicSystem(
            tax_rate=config.get("tax_rate", 0.3),  # Ensure tax rate is configurable
            redistribution_weight=config.get("redistribution_weight", 0.5)
        )
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        observation, reward, done, info = super().step(action)
        
        # Apply economic system logic after each step
        self.economic_system.apply_taxation(self.annotators)
        
        return observation, reward, done, info
