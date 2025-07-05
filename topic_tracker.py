# topic_tracker.py
import re
from collections import defaultdict, deque
from datetime import datetime

class TopicContextTracker:
    def __init__(self, max_history=20, debug=False):
        self.topic_history = deque(maxlen=max_history)
        self.topic_weights = defaultdict(float)
        self.last_updated = datetime.now()
        self.debug = debug

    def extract_keywords(self, text):
        # Basic keyword extraction (can be replaced with NLP tools later)
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = {"the", "is", "and", "to", "of", "a", "i", "it", "that", "this", "in", "on", "for", "with"}
        return [word for word in words if word not in stopwords and len(word) > 2]

    def update_context(self, user_input):
        keywords = self.extract_keywords(user_input)
        self.last_updated = datetime.now()

        for keyword in keywords:
            self.topic_weights[keyword] += 1.0

        self.topic_history.append((user_input, keywords))

        if self.debug:
            print(f"[TopicContextTracker] Updated with input: '{user_input}'")
            print(f"  Keywords: {keywords}")
            print(f"  Current weights: {dict(self.topic_weights)}")

    def add_input(self, user_input):
        self.update_context(user_input)

    def get_summary(self):
        return self.summarize_context()

    def get_current_topics(self, top_n=5):
        sorted_topics = sorted(self.topic_weights.items(), key=lambda item: item[1], reverse=True)
        return [topic for topic, weight in sorted_topics[:top_n]]

    def summarize_context(self):
        topics = self.get_current_topics()
        return f"Current conversation is focused on: {', '.join(topics)}." if topics else "No strong topics yet."

    def decay_weights(self, decay_rate=0.1):
        for topic in list(self.topic_weights):
            self.topic_weights[topic] -= decay_rate
            if self.topic_weights[topic] <= 0:
                del self.topic_weights[topic]

    def reset(self):
        self.topic_history.clear()
        self.topic_weights.clear()


# Example use case:
if __name__ == "__main__":
    tracker = TopicContextTracker(debug=True)
    tracker.update_context("I think Kohana needs better emotional depth.")
    tracker.update_context("She should also track topics like magic and robotics.")
    print(tracker.summarize_context())
