import random
from typing import Tuple, Union
from annotator_agent import AnnotatorAgent
import numpy as np


class MaliciousAnnotator(AnnotatorAgent):
    def __init__(self, annotator_id: int, config: dict):
        super().__init__(annotator_id, config)
        self.malicious = True

    def annotate_data(self, observation: np.ndarray) -> Tuple[Union[str, None], Union[Tuple, None], Union[str, None]]:
        """
        Generate a malicious annotation by picking a random incorrect label.
        
        Parameters:
        observation (np.ndarray): The data point to be annotated.
        
        Returns:
        Tuple containing the annotation, state, and action type.
        """
        if self.suspended:
            return None, None, None  # If the annotator is suspended, they do not provide an annotation

        data_complexity = self._calculate_data_complexity(observation)
        annotation_time = self._allocate_time(data_complexity)
        state = self._get_state(data_complexity, annotation_time)

        # Generate a malicious annotation by picking a random incorrect label
        incorrect_labels = [label for label in (self.categories + ['Unknown Disease', 'Malicious Disease']) if label != self.annotation]
        annotation = random.choice(incorrect_labels)

        self.time_budget -= annotation_time
        self.emotion['fatigue'] += annotation_time * 0.2

        self.annotation = annotation
        return annotation, state, 'malicious'

class RandomMaliciousAnnotator(AnnotatorAgent):
    def __init__(self, annotator_id: int, config: dict):
        super().__init__(annotator_id, config)
        self.malicious = True

    def annotate_data(self, observation: np.ndarray) -> Tuple[Union[str, None], Union[Tuple, None], Union[str, None]]:
        if self.suspended:
            return None, None, None
        
        data_complexity = self._calculate_data_complexity(observation)
        annotation_time = self._allocate_time(data_complexity)
        state = self._get_state(data_complexity, annotation_time)

        # Generate a random incorrect annotation
        incorrect_labels = [label for label in self.categories if label != self.annotation]
        annotation = random.choice(incorrect_labels)

        self.time_budget -= annotation_time
        self.emotion['fatigue'] += annotation_time * 0.2

        self.annotation = annotation
        return annotation, state, 'random_malicious'

class AlwaysIncorrectAnnotator(AnnotatorAgent):
    def __init__(self, annotator_id: int, config: dict, incorrect_label: str):
        super().__init__(annotator_id, config)
        self.malicious = True
        self.incorrect_label = incorrect_label

    def annotate_data(self, observation: np.ndarray) -> Tuple[Union[str, None], Union[Tuple, None], Union[str, None]]:
        if self.suspended:
            return None, None, None
        
        data_complexity = self._calculate_data_complexity(observation)
        annotation_time = self._allocate_time(data_complexity)
        state = self._get_state(data_complexity, annotation_time)

        # Always annotate with the specified incorrect label
        annotation = self.incorrect_label

        self.time_budget -= annotation_time
        self.emotion['fatigue'] += annotation_time * 0.2

        self.annotation = annotation
        return annotation, state, 'always_incorrect'

class PatternBasedMaliciousAnnotator(AnnotatorAgent):
    def __init__(self, annotator_id: int, config: dict):
        super().__init__(annotator_id, config)
        self.malicious = True

    def annotate_data(self, observation: np.ndarray) -> Tuple[Union[str, None], Union[Tuple, None], Union[str, None]]:
        if self.suspended:
            return None, None, None
        
        data_complexity = self._calculate_data_complexity(observation)
        annotation_time = self._allocate_time(data_complexity)
        state = self._get_state(data_complexity, annotation_time)

        # Choose an incorrect label based on data complexity
        incorrect_labels = [label for label in self.categories if label != self.annotation]
        if data_complexity > 0.7:
            annotation = random.choice(incorrect_labels)
        else:
            annotation = random.choice(self.categories)  # May include correct label for less complexity

        self.time_budget -= annotation_time
        self.emotion['fatigue'] += annotation_time * 0.2

        self.annotation = annotation
        return annotation, state, 'pattern_based_malicious'

class RandomStrategicMaliciousAnnotator(AnnotatorAgent):
    def __init__(self, annotator_id: int, config: dict):
        super().__init__(annotator_id, config)
        self.malicious = True

    def annotate_data(self, observation: np.ndarray) -> Tuple[Union[str, None], Union[Tuple, None], Union[str, None]]:
        if self.suspended:
            return None, None, None
        
        data_complexity = self._calculate_data_complexity(observation)
        annotation_time = self._allocate_time(data_complexity)
        state = self._get_state(data_complexity, annotation_time)

        # Combine random choice and strategic choice
        if random.random() > 0.5:
            # Random incorrect annotation
            incorrect_labels = [label for label in self.categories if label != self.annotation]
            annotation = random.choice(incorrect_labels)
        else:
            # Strategic incorrect annotation based on data complexity
            if data_complexity > 0.5:
                annotation = random.choice([label for label in self.categories if label != self.annotation])
            else:
                annotation = random.choice(self.categories)  # May include correct label for less complexity

        self.time_budget -= annotation_time
        self.emotion['fatigue'] += annotation_time * 0.2

        self.annotation = annotation
        return annotation, state, 'random_strategic_malicious'