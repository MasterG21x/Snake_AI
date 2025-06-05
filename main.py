from agent import DQNAgent
from app import Board   


from app import Board   

STATE_SIZE = 11
ACTION_SIZE = 3
EPISODES = 1000

agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

render_now = False        
FPS_PREVIEW = 10

for ep in range(EPISODES):
    game = Board(headless=not render_now)
    state, done = game.get_state(), False
    total_reward = 0.0

    while not done:
        action = agent.act(state)
        reward, done, _ = game.play_step(action)
        next_state = game.get_state()

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

        if render_now:
            game.render()
            game.clock.tick(FPS_PREVIEW)

    agent.decay_eps()
    print(f"Episode {ep:4d} | reward: {total_reward:6.1f} | ε = {agent.epsilon:.3f}")
    
    #Here u can change render handling
    if ep % 300 == 299:
        render_now = not render_now
