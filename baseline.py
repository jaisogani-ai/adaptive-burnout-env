from environment import ProductivityEnv
from grader import ProductivityGrader

def run_baseline(task_name):
    env = ProductivityEnv()
    state = env.reset()
    done = False
    trajectory = []

    while not done:
        stress = state.get("stress", 0.0)
        energy = state.get("energy", 0.0)
        motivation = state.get("motivation", 0.0)

        if stress > 0.7:
            action = 1
        elif energy < 0.3:
            action = 1
        elif motivation < 0.4:
            action = 2
        else:
            action = 0

        next_state, reward, done, info = env.step(action)
        step_data = {
            "obs": next_state,
        }
        trajectory.append(step_data)
        state = next_state

    grader = ProductivityGrader()
    if task_name == "easy":
        result = grader.grade_easy(trajectory)
    elif task_name == "medium":
        result = grader.grade_medium(trajectory)
    else:
        result = grader.grade_hard(trajectory)

    print(f"Task: {task_name} | Score: {result['score']} | Pass: {result['pass']}")

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_baseline(task)
