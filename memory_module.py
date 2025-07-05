# MemoryModule Class: Polished and Ready for Production

import sqlite3
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional

class MemoryModule:
    def __init__(self, db_path="memories.db", embedding_model="all-MiniLM-L6-v2", debug=False):
        self.db_path = db_path
        self.debug = debug
        self.embedding_model = SentenceTransformer(embedding_model)
        self.faiss_index = faiss.IndexFlatL2(384)
        self.memory_ids = []
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._initialize_database()
        self._load_faiss_index()
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=AutoModelForSequenceClassification.from_pretrained("./Local_Version2.0/cardiffnlp-twitter-roberta-base-sentiment-latest"),
            tokenizer=AutoTokenizer.from_pretrained("./Local_Version2.0/cardiffnlp-twitter-roberta-base-sentiment-latest"), device=-1  # Use CPU
        )


    def _initialize_database(self):
        """Create necessary database tables."""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY,
            content TEXT,
            context TEXT,
            category TEXT,
            embedding BLOB,
            timestamp REAL,
            importance_score REAL,
            emotional_impact REAL,
            sentiment REAL,
            related_memory_ids TEXT
        )
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY,
            person_name TEXT,
            context TEXT,
            user_id INTEGER UNIQUE,
            sentiment REAL,
            trust_level REAL,
            associated_memory_ids TEXT,
            timestamp Real
        )
        """)

        self.conn.commit()

    def _load_faiss_index(self):
        self.cursor.execute("SELECT id, embedding FROM memories")
        rows = self.cursor.fetchall()
        if rows:
            embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
            self.memory_ids = [row[0] for row in rows]
            self.faiss_index.add(np.array(embeddings))
        if self.debug:
            print(f"FAISS index initialized with {len(rows)} memories.")

    def add_memory(self, content, context="default", importance_score=0.5):
        sentiment, confidence = self.analyze_sentiment(content)
        importance_score += confidence * 0.1
        embedding = self.embedding_model.encode(content, convert_to_numpy=True)
        self.cursor.execute(
            "INSERT INTO memories (content, context, embedding, timestamp, importance_score, sentiment, emotional_impact) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (content, context, embedding.tobytes(), time.time(), importance_score, sentiment, confidence)
        )
        memory_id = self.cursor.lastrowid
        self.conn.commit()
        self.faiss_index.add(np.array([embedding]))
        self.memory_ids.append(memory_id)
        if self.debug:
            print(f"Added memory ID {memory_id} in context '{context}': {content}")

            

    def retrieve_relevant_memories(self, query, context="default", top_k=2):
        # Generate the query embedding and normalize it
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize

        # Search for the most relevant memories
        distances, indices = self.faiss_index.search(np.array([query_embedding]), top_k)
        results = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.memory_ids):
                try:
                    memory_id = self.memory_ids[idx]
                    self.cursor.execute("""
                        SELECT content, context, importance_score, timestamp
                        FROM memories WHERE id = ? AND context = ?
                    """, (memory_id, context))
                    result = self.cursor.fetchone()

                    if result and distance < 1e6:
                        results.append({
                            "id": memory_id,
                            "content": result[0],
                            "context": result[1],
                            "importance_score": result[2],
                            "distance": distance,
                            "timestamp": result[3],
                        })
                except Exception as e:
                    if self.debug:
                        print(f"Error retrieving memory ID {idx}: {e}")

        # (Optional: Pretty print if you want to see it in logs)
        #if self.debug:
            #for i, memory in enumerate(results, 1):
                #print(f"Memory {i}: {memory['content']} (Context: {memory['context']}, Importance: {memory['importance_score']:.2f}, Distance: {memory['distance']:.2f}, Timestamp: {memory['timestamp']})")

        return results

    def get_last_journal(self) -> Optional[str]:
        """
        Return the content of the most recent memory with context 'journal',
        or None if there aren’t any.
        """
        self.cursor.execute(
            "SELECT content FROM memories WHERE context = ?"
            " ORDER BY timestamp DESC LIMIT 1",
            ("journal",)
        )
        row = self.cursor.fetchone()
        return row[0] if row else None
    
    def add_core_memory(self, content, label="Core Memory", importance_score=0.95):
        """Add a memory marked as a 'core memory' for pivotal events."""
        formatted_content = f"[{label}] {content}"
        self.add_memory(
            content=formatted_content,
            context="core_memory",
            importance_score=importance_score
        )
        if self.debug:
            print(f"📌 Core memory saved: {label}")
        

        
    def delete_memory(self, memory_id):
        try:
            idx = self.memory_ids.index(memory_id)
            if idx < self.faiss_index.ntotal:
                self.faiss_index.remove_ids(np.array([idx]))
            self.memory_ids.pop(idx)
            self.cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self.conn.commit()
            if self.debug:
                print(f"Deleted memory ID {memory_id}.")
        except ValueError:
            if self.debug:
                print(f"Memory ID {memory_id} not found in FAISS index.")


    def update_memory(self, memory_id, new_content=None, new_importance_score=None):
        """Update a memory's content or importance score."""
        if new_content:
            embedding = self.embedding_model.encode(new_content, convert_to_numpy=True)
            idx = self.memory_ids.index(memory_id)
            self.faiss_index.reconstruct(idx)[:] = embedding
            self.cursor.execute("UPDATE memories SET content = ?, embedding = ? WHERE id = ?",
                                (new_content, embedding.tobytes(), memory_id))
        if new_importance_score:
            self.cursor.execute("UPDATE memories SET importance_score = ? WHERE id = ?", (new_importance_score, memory_id))
        self.conn.commit()
        if self.debug:
            print(f"Updated memory ID {memory_id}.")

    def close(self):
        self.cursor.close()
        self.conn.close()
        if self.debug:
            print("Database connection and cursor closed.")
    
                
            
    def calculate_importance(self, content, mode="default", related_memory_count=0, keywords=None):
        base_importance = 0.5
        if len(content) > 100:
            base_importance += 0.1
        base_importance += related_memory_count * 0.05
       
        if mode == "work_mode" and keywords and "task" in keywords:
            base_importance += 0.2
        elif mode == "game_mode" and keywords and "fun" in keywords:
            base_importance += 0.2
       
        return min(base_importance, 1.0)

    def chunk_text(self, text, chunk_size=512):
        words = text.split()
        for i in range(0, len(words), chunk_size):
            yield " ".join(words[i:i + chunk_size])

    def analyze_sentiment(self, content):
        """Analyze sentiment with chunking for long content."""
        try:
            chunks = self.chunk_text(content)
            sentiments, confidences = [], []
            for chunk in chunks:
                if self.debug:
                    print(f"Processing chunk: {chunk}")
                result = self.sentiment_pipeline(chunk)[0]
                if self.debug:
                    print(f"Sentiment result: {result}")
                sentiments.append({"LABEL_0": -1, "LABEL_1": 0, "LABEL_2": 1}[result["label"]])
                confidences.append(result["score"])
            # Average results
            avg_sentiment = sum(sentiments) / len(sentiments)
            avg_confidence = sum(confidences) / len(confidences)
            return avg_sentiment, avg_confidence
        except Exception as e:
            #if self.debug:
                #print(f"Sentiment analysis failed: {e}")
            return 0.0, 0.0
    
    def get_memory_by_id(self, memory_id):
        """Retrieve memory by its ID."""
        self.cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        return self.cursor.fetchone()

    
    def rebuild_faiss_index(self):
        self.faiss_index.reset()
        self.cursor.execute("SELECT id, embedding FROM memories")
        rows = self.cursor.fetchall()
        if rows:
            embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
            self.faiss_index.add(np.array(embeddings))
            self.memory_ids = [row[0] for row in rows]
        if self.debug:
            print(f"Rebuilt FAISS index with {len(rows)} memories.")
    

            
    def decay_memories(self, decay_rate=0.95, threshold=0.1):
        self.cursor.execute("""
            DELETE FROM memories
            WHERE importance_score < ?
            OR (importance_score * ? < ? AND timestamp < ?)
        """, (threshold, decay_rate, threshold, time.time() - 86400))
        self.conn.commit()
            
        
if __name__ == "__main__":
    memory_manager = MemoryModule(debug=True)

    memory_manager.add_memory("I went to the park and had a great time walking my dog.")
    relevant_memories = memory_manager.retrieve_relevant_memories("park", context="default")
    print("Relevant memories:", relevant_memories)
