import sqlite3
import time
import random

class GoalManager:
    def __init__(self, db_path="goals.db", debug=False):
        self.db_path = db_path
        self.debug = debug
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._initialize_database()
        self.preset_goals = [
            {"goal": "get more Kitsune-Bucks", "description": "Increase the amount of Kitsune-Bucks earned."},
            {"goal": "improve conversation skills", "description": "Enhance the ability to hold meaningful conversations."},
            {"goal": "learn new facts", "description": "Acquire new knowledge and facts."}
        ]
        self._load_preset_goals()

    def _initialize_database(self):
        """Create necessary database tables."""
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY,
            goal TEXT,
            description TEXT,
            status TEXT,
            progress REAL,
            timestamp REAL
        )
        """)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS subgoals (
            id INTEGER PRIMARY KEY,
            goal_id INTEGER,
            subgoal TEXT,
            status TEXT,
            progress REAL,
            FOREIGN KEY(goal_id) REFERENCES goals(id)
        )
        """)
        self.conn.commit()

    def _load_preset_goals(self):
        """Load preset goals into the database."""
        for goal in self.preset_goals:
            self.cursor.execute("SELECT * FROM goals WHERE goal = ?", (goal["goal"],))
            if not self.cursor.fetchone():
                self.add_goal(goal["goal"], goal["description"], status="permanent")

    def add_goal(self, goal, description, status="active"):
        """Add a new goal to the database."""
        self.cursor.execute(
            "INSERT INTO goals (goal, description, status, progress, timestamp) VALUES (?, ?, ?, ?, ?)",
            (goal, description, status, 0.0, time.time())
        )
        self.conn.commit()
        if self.debug:
            print(f"Added goal: {goal}")

    def update_goal_progress(self, goal_id, progress):
        """Update the progress of a goal."""
        self.cursor.execute("UPDATE goals SET progress = ? WHERE id = ?", (progress, goal_id))
        self.conn.commit()
        if self.debug:
            print(f"Updated progress for goal ID {goal_id} to {progress}")

    def complete_goal(self, goal_id):
        """Mark a goal as completed."""
        self.cursor.execute("UPDATE goals SET status = 'completed' WHERE id = ?", (goal_id,))
        self.conn.commit()
        if self.debug:
            print(f"Completed goal ID {goal_id}")

    def get_goals(self, status=None):
        """Retrieve goals from the database."""
        if status:
            self.cursor.execute("SELECT * FROM goals WHERE status = ? OR status = 'permanent'", (status,))
        else:
            self.cursor.execute("SELECT * FROM goals")
        return self.cursor.fetchall()
    

    def add_subgoal(self, goal_id, subgoal_text):
        self.cursor.execute(
            "INSERT INTO subgoals (goal_id, subgoal, status, progress) VALUES (?, ?, ?, ?)",
            (goal_id, subgoal_text, "active", 0.0)
        )
        self.conn.commit()
        if self.debug:
            print(f"Added subgoal to goal {goal_id}: {subgoal_text}")

    def get_subgoals(self, goal_id):
        self.cursor.execute("SELECT * FROM subgoals WHERE goal_id = ?", (goal_id,))
        return self.cursor.fetchall()

    def complete_subgoal(self, subgoal_id):
        self.cursor.execute("UPDATE subgoals SET status = 'completed', progress = 1.0 WHERE id = ?", (subgoal_id,))
        self.conn.commit()
        if self.debug:
            print(f"Subgoal {subgoal_id} marked as completed.")

    def get_active_tasks(self):
        self.cursor.execute("SELECT * FROM goals WHERE status = 'active'")
        goals = self.cursor.fetchall()
        self.cursor.execute("SELECT * FROM subgoals WHERE status = 'active'")
        subgoals = self.cursor.fetchall()
        return goals, subgoals


    def create_action_map(self, goal_id):
        """Create an action map for achieving a goal."""
        self.cursor.execute("SELECT goal, description FROM goals WHERE id = ?", (goal_id,))
        goal = self.cursor.fetchone()
        if not goal:
            if self.debug:
                print(f"Goal ID {goal_id} not found.")
            return None

        actions = [
            f"Research ways to achieve {goal[0]}",
            f"Break down {goal[0]} into smaller tasks",
            f"Set deadlines for tasks related to {goal[0]}",
            f"Track progress towards {goal[0]}",
            f"Review and adjust the plan for {goal[0]}"
        ]
        action_map = {"goal": goal[0], "description": goal[1], "actions": actions}
        if self.debug:
            print(f"Created action map for goal ID {goal_id}: {action_map}")
        return action_map

    def close(self):
        self.cursor.close()
        self.conn.close()
        if self.debug:
            print("Database connection closed.")

if __name__ == "__main__":
    goal_manager = GoalManager(debug=True)
    goal_manager.add_goal("learn Python", "Improve Python programming skills.")
    goals = goal_manager.get_goals()
    print("Goals:", goals)
    action_map = goal_manager.create_action_map(goals[0][0])
    print("Action Map:", action_map)
