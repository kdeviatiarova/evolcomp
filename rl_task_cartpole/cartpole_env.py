import math
import numpy as np
from typing import Optional
# from gym.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
# from gym import spaces, logger
from gymnasium import spaces, logger
import pygame
from pygame import gfxdraw


class CartPoleLeftRightEnv(CartPoleEnv):
    def __init__(self, env_config=None):
        super(CartPoleLeftRightEnv, self).__init__()
        # for speedup rendering increase this value
        self.metadata["render_fps"] = 500

        # set default angle threshold to 15°
        self.theta_threshold_radians = 15 * 2 * math.pi / 360

        # max x values
        self.x_threshold = 2.4
        # for _goal_reward function
        self.x_reward_threshold = 0.4
        self.x_reward_interval = self.x_threshold - self.x_reward_threshold
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                # add one value for goal
                1.0
            ],
            dtype=np.float32,
        )
        # set low obs value for goal
        low = -high
        low[4] = 0.0
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # self.action_space = spaces.Discrete(2)

        # todo any setup for your reward function...
        #...

        self.time_limit = None
        if 'time_limit' in env_config:
            self.time_limit = env_config['time_limit']


    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        self.cur_step += 1
        # add goal extraction
        x, x_dot, theta, theta_dot, goal = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
                       force + self.polemass_length * theta_dot ** 2 * sintheta
               ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # lets add goal to our state
        self.state = (x, x_dot, theta, theta_dot, goal)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        # check for time limit
        time_done = False
        if self.time_limit is not None:
            if self.cur_step > self.time_limit:
                done = True
                time_done = True

        # TODO here you need to design your reward!!!
        if not done:
            if not done:
        # You can adjust the reward calculation based on your goal and the position of the Cart
                if goal == 0:
            # Example: Reward for balancing on the left side
                  reward = 1.0 - 0.1 * abs(x - (-2.4))
                else:
            # Example: Reward for balancing on the right side
                  reward = 1.0 - 0.1 * abs(x - 2.4)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            if time_done:
                reward = 0.0
            else:
                if goal == 0:
                    reward = -1.0 if x > 0 else -0.5
                else:
                    reward = -1.0 if x < 0 else -0.5
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        # TODO add render here if u want to visualize all envs with rllib
        self.render()

        return np.array(self.state, dtype=np.float32), reward, done, None, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = True,
            options: Optional[dict] = None,
    ):
        super(CartPoleEnv, self).reset()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(5,))
        # set initial position exactly to 0.0
        # self.state[0] = 0.0
        # random goal selection and add to our observation
        self.state[4] = self.np_random.integers(2)
        self.cur_goal = self.state[4]
        self.cur_step = 0
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}


    # add rendering of goal as red line
    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        # draw goal
        goal_coords = [(0, 100), (0, 150), (5, 150), (5, 100)] if self.cur_goal == 0 else \
            [(screen_width-6, 100), (screen_width-6, 150), (screen_width-1, 150), (screen_width-1, 100)]
        gfxdraw.filled_polygon(self.surf, goal_coords, (255, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen


# if you want to debug env:
if __name__ == "__main__":
    env = CartPoleLeftRightEnv(env_config={})
    for _ in range(10):
        obs = env.reset()
        goal = obs[4]
        score = 0.0
        ep_len = 0
        for _ in range(300):
            obs, reward, done, info = env.step(env.action_space.sample())
            env.render()
            score += reward
            ep_len += 1
            if done:
                break
        print(ep_len, " ", score, "goal", goal)
