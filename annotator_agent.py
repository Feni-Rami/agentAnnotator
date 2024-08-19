import numpy as np
import random
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Union

class AnnotatorAgent:
    def __init__(self, annotator_id: str, config: Dict):
        self.annotator_id = annotator_id
        self.time_budget = np.random.uniform(*config.get("time_budget_range", (0.8, 1.2)))
        self.experience = config.get("experience", {'positive': 0, 'negative': 0})
        self.reputation = config.get("reputation", 1.0)
        self.income = 0.0  # New attribute for economic simulation
        self.resources = {'time': self.time_budget, 'accuracy': self.reputation}  # Resources for trading
        self.tax = 0.0  # Tax to be applied based on performance
        self.emotion = {
            'confidence': np.random.uniform(*config.get("confidence_range", (0.2, 0.8))),
            'stress': np.random.uniform(*config.get("stress_range", (0.2, 0.8))),
            'fatigue': np.random.uniform(*config.get("fatigue_range", (0.2, 0.8)))
        }
        self.categories = config.get("categories", ['Hypertension', 'Diabetes', 'Heart Disease', 'Kidney Disease', 'Liver Disease'])
        self.epsilon = 0.2
        self.alpha = 0.8
        self.gamma = 0.7
        self.experience_replay = deque(maxlen=config.get("experience_replay_size", 1000))
        self.Q_values = {}
        self.data = config.get("data")
        self.truth = config.get("truth")
        self.primary_model = RandomForestClassifier()
        self.secondary_model = LogisticRegression()
        self.train_models()

        self.annotation = None
        self.penalty_count = 0
        self.suspended = False
        self.reward = 0
        self.accuracy = 0

    def train_models(self):
        X_train = self.data
        y_train = self.truth
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.primary_model.fit(X_train, y_train)
        self.secondary_model.fit(X_train_scaled, y_train)

    def calculate_income(self, accuracy, complexity, improvement):
        base_income = accuracy * complexity * 100
        bonus = improvement * 50
        self.income = base_income + bonus

    def trade_resources(self, other_agent, resource_offer, resource_request):
        if self.resources['time'] >= resource_offer and other_agent.resources['accuracy'] >= resource_request:
            self.resources['time'] -= resource_offer
            other_agent.resources['time'] += resource_offer
            self.resources['accuracy'] += resource_request
            other_agent.resources['accuracy'] -= resource_request

    def apply_tax(self, tax_rate):
        self.tax = self.income * tax_rate
        self.income -= self.tax

    def annotate_data(self, observation: np.ndarray) -> Union[Tuple[str, Tuple, str], None]:
        if self.suspended:
            return None
        data_complexity = self._calculate_data_complexity(observation)
        annotation_time = self._allocate_time(data_complexity)
        state = self._get_state(data_complexity, annotation_time)
        action = self._choose_annotation_strategy(state)
        annotation = self._generate_annotation(action, observation)
        self.time_budget -= annotation_time
        self.emotion['fatigue'] += annotation_time * 0.2  
        self.annotation = annotation
        return annotation, state, action

    def _get_state(self, data_complexity: float, annotation_time: float) -> Tuple:
        economic_state = (self.income, self.tax)  # Include economic factors in state
        return (data_complexity, annotation_time) + tuple(self.emotion.values()) + economic_state

    def receive_feedback(self, correct: bool, state: Tuple, action: str, next_state: Tuple):
        self._update_internal_state_feedback(correct)
        self._update_q_values(state, action, self.reward, next_state)

    def _update_q_values(self, state: Tuple, action: str, reward: float, next_state: Tuple):
        current_q = self.Q_values.get((state, action), 0)
        future_q = max(self.Q_values.get((next_state, next_action), 0) for next_action in ['confirmation', 'proofreading'])
        updated_q = current_q + self.alpha * (reward + self.gamma * future_q - current_q)
        self.Q_values[(state, action)] = updated_q

    def _update_internal_state_feedback(self, correct: bool):
        if correct:
            self.experience['positive'] += 1
            self.reputation += 0.1
            self.reward += 10
            self.emotion['confidence'] = min(1.0, self.emotion['confidence'] + 0.1)
        else:
            self.experience['negative'] += 1
            self.reputation -= 0.1
            self.reward -= 10
            self.emotion['confidence'] = max(0.0, self.emotion['confidence'] - 0.1)
            self.emotion['stress'] = min(1.0, self.emotion['stress'] + 0.1)
        total_annotations = self.experience['positive'] + self.experience['negative']
        self.accuracy = self.experience['positive'] / total_annotations if total_annotations > 0 else 0.0

    def _choose_annotation_strategy(self, state: Tuple) -> str:
        economic_factor = self.income - self.tax
        if random.random() < self.epsilon:
            return random.choice(['confirmation', 'proofreading'])
        else:
            q_values = {strategy: self.Q_values.get((state, strategy), 0) for strategy in ['confirmation', 'proofreading']}
            if economic_factor < 0:
                return 'conservation'  # Hypothetical strategy for saving resources
            else:
                return max(q_values, key=q_values.get)

    def _generate_annotation(self, strategy: str, observation: np.ndarray) -> str:
        if strategy == 'confirmation':
            return self._confirm_annotation(observation)
        elif strategy == 'proofreading':
            return self._proofread_annotation(observation)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _confirm_annotation(self, observation: np.ndarray) -> str:
        prediction, confidence = self._model_predict_with_confidence(observation)
        if confidence > 0.8:
            return prediction
        else:
            secondary_prediction = self._secondary_model_predict(observation)
            secondary_confidence = self._secondary_model_predict_proba(observation)
            if secondary_confidence < 0.8:
                return self._choose_based_on_historical_distribution()
            else:
                return secondary_prediction

    def _secondary_model_predict_proba(self, observation: np.ndarray) -> float:
        return max(self.secondary_model.predict_proba([observation])[0])

    def _choose_based_on_historical_distribution(self) -> str:
        frequency_distribution = self._historical_frequency_distribution()
        return np.random.choice(self.categories, p=frequency_distribution)
    
    def _proofread_annotation(self, observation: np.ndarray) -> str:
        return self._secondary_model_predict(observation)

    def _model_predict_with_confidence(self, observation: np.ndarray) -> Tuple[str, float]:
        predicted_category = self.primary_model.predict([observation])[0]
        confidence = max(self.primary_model.predict_proba([observation])[0])
        return predicted_category, confidence

    def _secondary_model_predict(self, observation: np.ndarray) -> str:
        return self.secondary_model.predict([observation])[0]

    def _historical_frequency_distribution(self) -> np.ndarray:
        category_counts = {category: 0 for category in self.categories}
        for _, _, _, next_state in self.experience_replay:
            if 'annotation' in next_state:
                annotation = next_state['annotation']
                if annotation in category_counts:
                    category_counts[annotation] += 1
        counts_array = np.array([category_counts[category] for category in self.categories])
        if np.sum(counts_array) == 0:
            frequency_distribution = np.ones(len(self.categories)) / len(self.categories)
        else:
            frequency_distribution = counts_array / np.sum(counts_array)
        return frequency_distribution

    def _calculate_data_complexity(self, observation: np.ndarray) -> float:
        if len(observation) == 0:
            return 1.0

        normalized_observation = (observation - np.min(observation)) / (np.ptp(observation) + 1e-6)
        std_dev = np.std(normalized_observation)
        hist, bin_edges = np.histogram(normalized_observation, bins=10, density=True)
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
