import numpy as np
import random

class NOCEnv:
    def __init__(self):
        self.buffer_size = {'CPU': 100, 'IO': 100}
        self.arbiter_weights = {'CPU': 0.5, 'IO': 0.5}
        self.operating_frequency = 1.0
        self.power_threshold = 1.2
        self.min_latency = 10
        self.max_bandwidth = 800  # Calculated as Bytes per second
        self.current_power_usage = 0.0
        self.last_power_usage = 0.0  # For hysteresis in throttling
        self.max_buffer_size = {'CPU': 200, 'IO': 200}
        self.data_width = 32  # Bytes per transfer
        self.transaction_log = []
        self.time = 0
        self.throttling = False
        self.throttling_counter = 0
        
        # Simulated addresses for reads and writes
        self.address_space = ['Addr1', 'Addr2', 'Addr3', 'Addr4', 'Addr5']
        self.read_latency = {addr: 1 for addr in self.address_space}
        self.write_latency = {addr: 2 for addr in self.address_space}

    def _get_state(self):
        # Calculate average latencies and buffer occupancy
        avg_read_latency = np.mean(list(self.read_latency.values()))
        avg_write_latency = np.mean(list(self.write_latency.values()))
        avg_latency = np.mean([avg_read_latency, avg_write_latency])
        buffer_occupancy = np.mean([self.buffer_size[k] / self.max_buffer_size[k] for k in self.buffer_size])
        return np.array([
            avg_latency,  # Average latency
            self.calculate_bandwidth(),  # Current bandwidth
            buffer_occupancy,  # Buffer occupancy percentage
            float(self.throttling)  # Throttling activity as float
        ])
        
    def step(self, action):
        self.apply_actions(action)
        self.generate_traffic()
        self.update_latencies()
        self.update_metrics()
        self.throttle()
        reward = self.calculate_reward()
        done = self.check_if_done()
        self.time += 1
        return self._get_state(), reward, done
    
    def update_latencies(self):
        """Adjust latencies based on operating frequency and actual operations."""
        for addr in self.read_latency:
            # Adjust latency based on the inverse of the operating frequency
            self.read_latency[addr] = max(5, int(100 / self.operating_frequency))
        for addr in self.write_latency:
            self.write_latency[addr] = max(3, int(80 / self.operating_frequency))

    def generate_traffic(self):
        for addr in self.address_space:
            operation_type = random.choice(['read', 'write'])
            # More varied and potentially impactful changes
            data = np.random.bytes(4).hex()
            latency = random.randint(5, 20)  # Wider range
            if operation_type == 'read':
                self.read_latency[addr] = latency
            else:
                self.write_latency[addr] = latency

    @property
    def latency(self):
        # Assuming avg_latency is calculated somewhere in the environment
        return np.mean(list(self.read_latency.values()) + list(self.write_latency.values()))

    @property
    def bandwidth(self):
        return self.calculate_bandwidth()

    @property
    def buffer_occupancy(self):
        total_capacity = sum(self.max_buffer_size.values())
        total_used = sum(self.buffer_size.values())
        return total_used / total_capacity if total_capacity > 0 else 0

    def print_transactions(self):
        print("Timestamp,TxnType,Data (32B)")
        for record in self.transaction_log:
            print(f"{record[0]},{record[1]} {record[2]},{record[3]}")

    def get_buffer_occupancy(self, buffer_id):
        """Return current buffer occupancy."""
        return self.buffer_size.get(buffer_id, 0)

    def get_arbrates(self, agent_type):
        """Return current arbitration rates."""
        return self.arbiter_weights.get(agent_type, 0)

    def set_max_buffer_size(self, buffer_id, num_entries):
        """Set the maximum buffer size for specified buffer."""
        self.max_buffer_size[buffer_id] = num_entries

    def set_arbiter_weights(self, agent_type, weight):
        """Adjust arbiter weights ensuring total does not exceed 1."""
        self.arbiter_weights[agent_type] = weight
        self.arbiter_weights['IO' if agent_type == 'CPU' else 'CPU'] = 1 - weight
    
    def throttle(self):
        """Throttle operating frequency based on power usage hysteresis."""
        if self.current_power_usage > self.power_threshold and not self.throttling:
            self.operating_frequency *= 0.9  # Apply the throttling action
            self.throttling = True
            self.throttling_counter += 1  # Increment the throttling counter since action is taken
        elif self.current_power_usage < self.last_power_usage:
            self.throttling = False

    def apply_actions(self, action):
        self.buffer_size['CPU'] = max(0, min(self.max_buffer_size['CPU'], self.buffer_size['CPU'] + action['adjust_cpu_buffer']))
        self.buffer_size['IO'] = max(0, min(self.max_buffer_size['IO'], self.buffer_size['IO'] + action['adjust_io_buffer']))
        self.arbiter_weights['CPU'] = max(0, min(1, self.arbiter_weights['CPU'] + action['adjust_cpu_weight']))
        self.arbiter_weights['IO'] = 1 - self.arbiter_weights['CPU']  # Ensure total is always 1
        self.operating_frequency = max(0.1, min(2, self.operating_frequency + action['adjust_frequency']))

    def update_metrics(self):
        """Update metrics related to bandwidth and power usage."""
        self.last_power_usage = self.current_power_usage
        self.current_power_usage = self.operating_frequency * (self.buffer_size['CPU'] + self.buffer_size['IO']) / 100
        self.total_bandwidth = self.calculate_bandwidth()

    def calculate_bandwidth(self):
        """Calculate the total bandwidth used based on buffer sizes and data width."""
        total_data_transferred = (self.buffer_size['CPU'] + self.buffer_size['IO']) * self.data_width
        return total_data_transferred * self.operating_frequency / 1000  # Convert to Bytes/sec

    def calculate_reward(self):
        reward = 0
        
        # 1. Latency Reward/Penalty
        avg_read_latency = np.mean(list(self.read_latency.values()))
        avg_write_latency = np.mean(list(self.write_latency.values()))
        avg_latency = np.mean([avg_read_latency, avg_write_latency])
        
        # Reward if latency is below the minimum target, penalize if above
        reward += 1 if avg_latency <= self.min_latency else -1

        # 2. Bandwidth Reward
        total_bandwidth = self.calculate_bandwidth()
        reward += 1 if total_bandwidth >= 0.95 * self.max_bandwidth else -1

        # 3. Buffer Occupancy Reward
        buffer_occupancy = np.mean([self.buffer_size[k] / self.max_buffer_size[k] for k in self.buffer_size])
        reward += 1 if 0.85 <= buffer_occupancy <= 0.95 else -1

        # 4. Throttling Penalty: Simplify to only penalize if over a threshold
        throttling_ratio = self.throttling_counter / self.time if self.time > 0 else 0
        reward -= 1 if throttling_ratio > 0.05 else 0

        return reward

    def check_if_done(self):
        return self.current_power_usage > self.power_threshold

    def reset(self):
        self.buffer_size = {'CPU': 100, 'IO': 100}
        self.arbiter_weights = {'CPU': 0.5, 'IO': 0.5}
        self.operating_frequency = 1.0
        self.current_power_usage = 0.0
        self.last_power_usage = 0.0
        self.read_latency = {addr: 0 for addr in self.address_space}
        self.write_latency = {addr: 0 for addr in self.address_space}
        self.throttling = False
        self.throttling_counter = 0
        self.transaction_log.clear()
        self.time = 0
        return self._get_state()