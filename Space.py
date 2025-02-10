import logging

import numpy as np
import gymnasium as gym
from sklearn.neighbors import KDTree

from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver

logging.basicConfig(filename='done_logs.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

class SpaceEnv(gym.Env):
    def __init__(self, num_satellites=9, time_step=60, max_steps=1000):
        self.num_satellites = num_satellites
        self.time_step = time_step
        self.max_steps = max_steps
        self.current_step = 0

        self.observation_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, 0] * self.num_satellites, dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1] * self.num_satellites, dtype=np.float32)
        )

        act_dim = self.num_satellites * 3
        self.action_space = gym.spaces.Box(low=-0.001, high=0.001, shape=(act_dim,), dtype=np.float32)

        self.orbits = []
        self.satellites_states = []

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        
        self.orbits = []
        self.satellites_states = []

        epoch = Time.now()

        for _ in range(self.num_satellites):
            alt = np.random.uniform(300, 700) * u.km
            inc = np.random.uniform(0, 180) * u.deg
            raan = np.random.uniform(0, 360) * u.deg
            arglat = np.random.uniform(0, 360) * u.deg

            orbit = Orbit.circular(
                Earth, alt=alt, inc=inc, raan=raan, arglat=arglat, epoch=epoch
            )

            self.orbits.append(orbit)

            r_vec = orbit.r.to(u.km).value
            v_vec = orbit.v.to(u.km / u.s).value
            fuel = 100

            sat_state = np.concatenate([r_vec, v_vec, [fuel]])
            self.satellites_states.append(sat_state)
        
        return self.__get_obs(), {}


    def step(self, action):
        self.current_step += 1

        max_fuel = float('-inf')
        min_fuel = float('inf')

        max_dv_norm = float('-inf')
        min_dv_norm = float('inf')

        max_ecc = float('-inf')
        min_ecc = float('inf')

        max_sma = float('-inf')
        min_sma = float('inf')

        for i in range(self.num_satellites):
            dv = action[i*3:(i+1)*3]
            dv_km_s = dv * u.km / u.s
            dv_norm = np.linalg.norm(dv_km_s.value)

            isp = 300
            g0 = 9.81

            orbit = self.orbits[i]

            fuel_left = self.satellites_states[i][6]

            if fuel_left == 0:
                new_orbit = orbit.propagate(self.time_step * u.s)
            else:
                man = Maneuver.impulse(dv_km_s)

                new_orbit = orbit.apply_maneuver(man)
                new_orbit = new_orbit.propagate(self.time_step * u.s)

                fuel_burn = dv_norm / (isp * g0) * 1000
                fuel_left = self.satellites_states[i][6] - fuel_burn
                fuel_left = max(fuel_left, 0.0)

            self.orbits[i] = new_orbit

            r_vec = new_orbit.r.to(u.km).value
            v_vec = new_orbit.v.to(u.km / u.s).value

            self.satellites_states[i] = np.concatenate([r_vec, v_vec, [fuel_left]])

            max_fuel = max(max_fuel, fuel_left)
            min_fuel = min(min_fuel, fuel_left)

            max_dv_norm = max(max_dv_norm, dv_norm)
            min_dv_norm = min(min_dv_norm, dv_norm)

            ecc = new_orbit.ecc.value
            sma = new_orbit.a.to(u.km).value

            max_ecc = max(max_ecc, ecc)
            min_ecc = min(min_ecc, ecc)

            max_sma = max(sma, max_sma)
            min_sma = min(sma, min_sma)
        
        reward, done = self._compute_reward_and_done(action)

        truncated = self.current_step >= self.max_steps

        obs = self.__get_obs()
        info = {
            "dv_avg": (max_dv_norm + min_dv_norm) / 2,
            "ecc_avg": (max_ecc + min_ecc) / 2,
            "sma_avg": (max_sma + min_sma) / 2,
            "fuel_avg": (max_fuel + min_fuel) / 2
        }

        return obs, reward, done, truncated, info

    def _compute_reward_and_done(self, action):
        reward = 0.0
        done = False

        eci = [xyz[0:3] for xyz in self.satellites_states]

        kd_tree = KDTree(eci)
        pairs = kd_tree.query_radius(eci, r=float('inf'), return_distance=True)

        for i, (n, distances) in enumerate(zip(*pairs)):
            for j, dist in zip(n, distances):
                if i < j:
                    # 1. Collision Avoidance (Penalty)
                    if dist < 1.0:
                        reward -= 10000000.0
                        done = True
                        logging.info(f"Satellite collission - Distances: {dist}")
                    elif dist < 3.0:
                        reward -= 100.0 * (3.0 - dist)
                    else:
                        ideal_dist = 100.0
                        reward += 10.0 * (1.0 - np.exp(-0.01 * (dist - ideal_dist)**2))
            
            # 2. Circular Orbit Reward
            ecc = self.orbits[i].ecc.value
            if ecc >= 1.0:
                reward -= 300.0 * ecc ** 2
            else:
                reward += 300.0 * np.exp(-10.0 * ecc)

            # 3. Semi-Major Axis (SMA) Penalty
            sma = max(self.orbits[i].a.to(u.km).value, 1.0)

            lower_bound = 6578  # 200 km altitude
            upper_bound = 8278  # 1900 km altitude
            
            if sma < lower_bound or sma > upper_bound:
                reward -= 3000.0
            else:
                reward += 300.0
            
            # 4. Velocity Magnitude Penalty
            v_orbit = np.sqrt(3.986e5 / sma)
            v_mag = np.linalg.norm(self.satellites_states[i][3:6])
            reward -= 10.0 * ((v_mag - v_orbit) / v_orbit) ** 2
            reward += 150.0 * np.exp(-10.0 * ((v_mag - v_orbit) / v_orbit) ** 2)

            # 5. Fuel Efficiency Reward
            fuel_left = self.satellites_states[i][6]
            reward += 5.0 * np.log(1 + (fuel_left / 100.0))
            reward -= 5.0 * ((fuel_left / 100.0) ** 2)

            if fuel_left < 80.0:
                reward -= 20.0 * (1.0 - (fuel_left / 100.0)) ** 2

            # 6. Smooth Actions (Penalty for abrupt maneuvers)
            dv_norm = np.linalg.norm(action[i*3:(i+1)*3])
            reward -= 0.001 * dv_norm * (1.0 + ecc**2)
            reward += 1.0 * np.exp(-200.0 * dv_norm ** 2)
                
        return reward, done

    def __get_obs(self):
        obs = []

        r_norm = 7000.0
        v_norm = np.sqrt(3.986e5 / 7000.0)
        f_norm = 100.0

        for sat_state in self.satellites_states:
            r_vec = sat_state[0:3] / r_norm
            v_vec = sat_state[3:6] / v_norm
            fuel = sat_state[6] / f_norm

            obs.append(np.concatenate([r_vec, v_vec, [fuel]]))
        return np.concatenate(obs).astype(np.float32)