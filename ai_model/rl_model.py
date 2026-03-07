"""
Reinforcement Learning Trading Agent - ตัดสินใจซื้อขายด้วย RL
"""
import os
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from config import AIConfig
from utils.logger import TradeLogger


class TradingEnvironment:
    """Trading environment for RL agent."""

    FEATURE_COLUMNS = [
        "close", "volume", "rsi", "macd", "macd_signal",
        "bb_upper", "bb_lower", "bb_pct", "ema_9", "ema_21",
        "volume_ratio",
    ]

    # Actions: 0 = HOLD, 1 = BUY, 2 = SELL
    HOLD = 0
    BUY = 1
    SELL = 2
    ACTION_SPACE = 3

    def __init__(self, df: pd.DataFrame, initial_balance: float = 100000.0,
                 commission: float = 0.0025):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission

        # Feature columns for state
        self.feature_columns = [col for col in self.FEATURE_COLUMNS if col in df.columns]
        self.state_size = len(self.feature_columns) + 2  # + position flag + unrealized pnl

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.buy_price = 0.0
        self.total_profit = 0.0
        self.trades = []
        self.done = False
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        if self.current_step >= len(self.df):
            return np.zeros(self.state_size)

        row = self.df.iloc[self.current_step]
        features = [row.get(col, 0) for col in self.feature_columns]

        # Normalize features
        price = row.get("close", 1)
        features = [f / price if abs(price) > 0 else 0 for f in features]

        # Add position info
        has_position = 1.0 if self.crypto_held > 0 else 0.0
        unrealized_pnl = 0.0
        if self.crypto_held > 0 and self.buy_price > 0:
            unrealized_pnl = (price - self.buy_price) / self.buy_price

        features.extend([has_position, unrealized_pnl])
        return np.array(features, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute one trading step."""
        if self.current_step >= len(self.df) - 1:
            self.done = True
            return self._get_state(), 0.0, True

        current_price = self.df.iloc[self.current_step]["close"]
        reward = 0.0

        if action == self.BUY and self.crypto_held == 0:
            # Buy
            cost = self.balance * 0.95  # Use 95% of balance
            commission = cost * self.commission
            self.crypto_held = (cost - commission) / current_price
            self.balance -= cost
            self.buy_price = current_price
            self.trades.append(("BUY", current_price, self.current_step))

        elif action == self.SELL and self.crypto_held > 0:
            # Sell
            revenue = self.crypto_held * current_price
            commission = revenue * self.commission
            net_revenue = revenue - commission
            profit = net_revenue - (self.initial_balance - self.balance)
            profit_pct = (current_price - self.buy_price) / self.buy_price
            reward = profit_pct * 100  # Reward based on profit percentage

            self.balance += net_revenue
            self.total_profit += profit
            self.crypto_held = 0.0
            self.trades.append(("SELL", current_price, self.current_step))

        elif action == self.HOLD:
            # Small penalty for holding to encourage action
            if self.crypto_held > 0:
                unrealized = (current_price - self.buy_price) / self.buy_price
                reward = unrealized * 0.1  # Small reward/penalty based on position
            else:
                reward = -0.01  # Tiny penalty for being idle

        self.current_step += 1
        next_state = self._get_state()
        self.done = self.current_step >= len(self.df) - 1

        return next_state, reward, self.done

    def get_portfolio_value(self) -> float:
        """Get current total portfolio value."""
        current_price = self.df.iloc[min(self.current_step, len(self.df) - 1)]["close"]
        return self.balance + self.crypto_held * current_price


class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions."""

    def __init__(self, state_size: int, action_size: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RLTradingAgent:
    """Reinforcement Learning agent for trading decisions."""

    def __init__(self, config: AIConfig, state_size: int = 13,
                 logger: Optional[TradeLogger] = None):
        self.config = config
        self.logger = logger or TradeLogger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = TradingEnvironment.ACTION_SPACE

        # Networks
        self.policy_net = DQNNetwork(state_size, self.action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.rl_learning_rate
        )
        self.criterion = nn.SmoothL1Loss()

        # Replay memory
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = config.rl_gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 10

    def select_action(self, state: np.ndarray, training: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train on a batch from replay memory."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(list(self.memory), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        self.policy_net.train()
        current_q = self.policy_net(states).gather(1, actions)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, df: pd.DataFrame, episodes: int = 0) -> dict:
        """Train the RL agent on historical data."""
        episodes = episodes or self.config.rl_episodes
        env = TradingEnvironment(df)
        self.state_size = env.state_size

        # Reinitialize networks with correct state size
        self.policy_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.config.rl_learning_rate
        )

        history = {"episode_rewards": [], "portfolio_values": []}
        self.logger.log_info(f"Starting RL training: {episodes} episodes")

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0

            while not env.done:
                action = self.select_action(state, training=True)
                next_state, reward, done = env.step(action)

                self.remember(state, action, reward, next_state, done)
                self.replay()

                state = next_state
                total_reward += reward

            # Update target network
            if (episode + 1) % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            portfolio_value = env.get_portfolio_value()
            history["episode_rewards"].append(total_reward)
            history["portfolio_values"].append(portfolio_value)

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(history["episode_rewards"][-50:])
                self.logger.log_info(
                    f"Episode {episode + 1}/{episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Portfolio: {portfolio_value:.2f} | "
                    f"Epsilon: {self.epsilon:.4f} | "
                    f"Trades: {len(env.trades)}"
                )

        self.save_model()
        self.logger.log_info("RL training complete")
        return history

    def decide(self, state: np.ndarray) -> dict:
        """Make a trading decision."""
        action = self.select_action(state, training=False)
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_t).cpu().numpy()[0]

        # Confidence based on Q-value difference
        q_sorted = sorted(q_values, reverse=True)
        confidence = (q_sorted[0] - q_sorted[1]) / (abs(q_sorted[0]) + 1e-8)
        confidence = min(max(confidence, 0), 1)

        return {
            "action": action,
            "action_name": action_names[action],
            "q_values": q_values.tolist(),
            "confidence": confidence,
        }

    def build_live_state(self, df: pd.DataFrame, has_position: bool = False,
                         entry_price: float = 0.0) -> Optional[np.ndarray]:
        """Build a live inference state from the latest indicator row."""
        if df is None or df.empty:
            return None

        row = df.iloc[-1]
        current_price = float(row.get("close", 0) or 0)
        if current_price <= 0:
            return None

        feature_columns = [
            col for col in TradingEnvironment.FEATURE_COLUMNS if col in df.columns
        ]
        features = []
        for col in feature_columns:
            try:
                value = float(row.get(col, 0) or 0)
            except (TypeError, ValueError):
                value = 0.0
            features.append(value / current_price if abs(current_price) > 0 else 0.0)

        unrealized_pnl = 0.0
        if has_position and entry_price > 0:
            unrealized_pnl = (current_price - entry_price) / entry_price

        features.extend([1.0 if has_position else 0.0, unrealized_pnl])
        state = np.array(features, dtype=np.float32)
        if state.size != self.state_size:
            return None
        return state

    def save_model(self):
        """Save model to disk."""
        os.makedirs(os.path.dirname(self.config.rl_model_path), exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "state_size": self.state_size,
            "action_size": self.action_size,
        }, self.config.rl_model_path)
        self.logger.log_info(f"RL model saved to {self.config.rl_model_path}")

    def load_model(self):
        """Load model from disk."""
        if not os.path.exists(self.config.rl_model_path):
            self.logger.log_info("No saved RL model found")
            return False

        checkpoint = torch.load(
            self.config.rl_model_path, map_location=self.device, weights_only=False
        )
        self.state_size = checkpoint["state_size"]
        self.action_size = checkpoint["action_size"]

        self.policy_net = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_net = DQNNetwork(self.state_size, self.action_size).to(self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)

        self.logger.log_info("RL model loaded")
        return True
