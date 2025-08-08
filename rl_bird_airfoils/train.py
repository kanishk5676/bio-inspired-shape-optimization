import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env

from env import AdvancedAirfoilEnvironment, AirfoilMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    algorithm: str = 'PPO'
    total_timesteps: int = 500000
    learning_rate: float = 3e-4
    n_parallel_envs: int = 8
    eval_freq: int = 10000
    eval_episodes: int = 10
    save_freq: int = 50000
    log_interval: int = 100
    
    # Environment parameters
    use_biological_constraints: bool = True
    multi_objective: bool = True
    adaptive_targets: bool = True
    reynolds_number: float = 1e6
    
    # Output directories
    output_dir: str = "./results"
    model_dir: str = "./models"
    log_dir: str = "./logs"


class AdvancedTrainingCallback(BaseCallback):
    """Advanced callback for comprehensive training monitoring"""
    
    def __init__(self, config: TrainingConfig, eval_env=None, verbose=1):
        super().__init__(verbose)
        self.config = config
        self.eval_env = eval_env
        
        # Tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.performance_metrics = []
        self.biological_scores = []
        self.convergence_rates = []
        
        # Best model tracking
        self.best_reward = -np.inf
        self.best_model_path = None
        
        # Curriculum tracking
        self.difficulty_progression = []
        self.success_rates = []
        
    def _on_step(self) -> bool:
        """Called after each step"""
        # Collect episode data when episode ends
        if self.locals.get('dones', [False])[0]:  # Episode finished
            if 'episode' in self.locals:
                episode_reward = self.locals['episode']['r']
                episode_length = self.locals['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Extract additional metrics from info
                if 'infos' in self.locals and len(self.locals['infos']) > 0:
                    info = self.locals['infos'][0]
                    
                    if 'performance_metrics' in info:
                        metrics = info['performance_metrics']
                        self.performance_metrics.append({
                            'cl_cd_ratio': metrics.cl_cd_ratio,
                            'cl': metrics.cl,
                            'cd': metrics.cd,
                            'stall_angle': metrics.stall_angle
                        })
                    
                    if 'biological_plausibility' in info:
                        self.biological_scores.append(info['biological_plausibility'])
                    
                    if 'difficulty_level' in info:
                        self.difficulty_progression.append(info['difficulty_level'])
                    
                    if 'convergence_rate' in info:
                        self.convergence_rates.append(info['convergence_rate'])
        
        # Periodic evaluation
        if self.num_timesteps % self.config.eval_freq == 0 and self.eval_env is not None:
            self._evaluate_model()
        
        # Save model periodically
        if self.num_timesteps % self.config.save_freq == 0:
            self._save_checkpoint()
        
        return True
    
    def _evaluate_model(self):
        """Evaluate current model performance"""
        if self.eval_env is None:
            return
        
        logger.info(f"Evaluating model at timestep {self.num_timesteps}")
        
        eval_rewards = []
        eval_metrics = []
        eval_bio_scores = []
        
        for _ in range(self.config.eval_episodes):
            obs = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            
            # Collect final metrics
            if 'performance_metrics' in info:
                metrics = info['performance_metrics']
                eval_metrics.append({
                    'cl_cd_ratio': metrics.cl_cd_ratio,
                    'cl': metrics.cl,
                    'cd': metrics.cd,
                    'stall_angle': metrics.stall_angle
                })
            
            if 'biological_plausibility' in info:
                eval_bio_scores.append(info['biological_plausibility'])
        
        # Calculate statistics
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        
        if eval_metrics:
            mean_ld = np.mean([m['cl_cd_ratio'] for m in eval_metrics])
            mean_cl = np.mean([m['cl'] for m in eval_metrics])
            mean_cd = np.mean([m['cd'] for m in eval_metrics])
        else:
            mean_ld = mean_cl = mean_cd = 0
        
        mean_bio = np.mean(eval_bio_scores) if eval_bio_scores else 0
        
        # Log results
        logger.info(f"Evaluation Results:")
        logger.info(f"  Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
        logger.info(f"  Mean L/D: {mean_ld:.1f}")
        logger.info(f"  Mean CL: {mean_cl:.3f}")
        logger.info(f"  Mean CD: {mean_cd:.4f}")
        logger.info(f"  Biological Score: {mean_bio:.3f}")
        
        # Save best model
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.best_model_path = os.path.join(
                self.config.model_dir, 
                f"best_model_reward_{mean_reward:.3f}_step_{self.num_timesteps}.zip"
            )
            self.model.save(self.best_model_path)
            logger.info(f"New best model saved: {self.best_model_path}")
        
        # Store evaluation data
        eval_data = {
            'timestep': self.num_timesteps,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_ld': mean_ld,
            'mean_cl': mean_cl,
            'mean_cd': mean_cd,
            'mean_biological_score': mean_bio,
            'individual_rewards': eval_rewards,
            'individual_metrics': eval_metrics
        }
        
        # Save evaluation data
        eval_file = os.path.join(self.config.log_dir, 'evaluations.json')
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                all_evals = json.load(f)
        else:
            all_evals = []
        
        all_evals.append(eval_data)
        
        with open(eval_file, 'w') as f:
            json.dump(all_evals, f, indent=2)
    
    def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(
            self.config.model_dir,
            f"checkpoint_step_{self.num_timesteps}.zip"
        )
        self.model.save(checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def get_training_statistics(self):
        """Get comprehensive training statistics"""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths, 
            'performance_metrics': self.performance_metrics,
            'biological_scores': self.biological_scores,
            'difficulty_progression': self.difficulty_progression,
            'convergence_rates': self.convergence_rates,
            'best_reward': self.best_reward,
            'best_model_path': self.best_model_path
        }


class BioinspiredTrainer:
    """Main trainer class for bio-inspired airfoil optimization"""
    
    def __init__(self, config: TrainingConfig, bird_data_path: str):
        self.config = config
        self.bird_data_path = bird_data_path
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Load and preprocess bird data
        self.bird_data = self._load_bird_data()
        
        # Initialize environments
        self.train_env = None
        self.eval_env = None
        self.model = None
        self.callback = None
        
    def _load_bird_data(self) -> pd.DataFrame:
        """Load and preprocess bird morphology data"""
        try:
            df = pd.read_csv(self.bird_data_path, encoding='ISO-8859-1')
            logger.info(f"Loaded {len(df)} bird species from {self.bird_data_path}")
        except Exception as e:
            logger.warning(f"Could not load bird data: {e}, generating sample data")
            df = self._generate_sample_bird_data()
        
        # Required columns for airfoil optimization
        required_cols = ['species', 'Wing.Length', 'Secondary1', 'Kipps.Distance', 'Hand-Wing.Index']
        
        # Handle missing columns
        for col in required_cols:
            if col not in df.columns:
                if col == 'species':
                    df['species'] = [f"Species_{i}" for i in range(len(df))]
                else:
                    logger.warning(f"Missing column {col}, using default values")
                    df[col] = np.random.uniform(50, 500, len(df))  # Reasonable default range
        
        # Clean data
        df_clean = df[required_cols].dropna()
        
        # Add missing Tail.Length if not present
        if 'Tail.Length' not in df_clean.columns:
            df_clean['Tail.Length'] = df_clean['Wing.Length'] * np.random.uniform(0.3, 0.8, len(df_clean))
        
        # Filter reasonable ranges
        df_clean = df_clean[
            (df_clean['Wing.Length'] > 20) & (df_clean['Wing.Length'] < 1000) &
            (df_clean['Secondary1'] > 10) & (df_clean['Secondary1'] < 600) &
            (df_clean['Kipps.Distance'] > 5) & (df_clean['Kipps.Distance'] < 500)
        ].reset_index(drop=True)
        
        logger.info(f"Using {len(df_clean)} species with complete morphological data")
        
        # Calculate derived features
        df_clean['aspect_ratio'] = df_clean['Wing.Length'] / df_clean['Secondary1']
        df_clean['kipps_ratio'] = df_clean['Kipps.Distance'] / df_clean['Wing.Length']
        df_clean['hand_wing_ratio'] = df_clean['Hand-Wing.Index'] / 100.0
        
        return df_clean
    
    def _generate_sample_bird_data(self) -> pd.DataFrame:
        """Generate realistic sample bird data"""
        np.random.seed(42)
        
        # Bird archetypes with realistic morphology
        archetypes = {
            'Hummingbird': {'wl': (40, 70), 's1': (20, 40), 'kd': (15, 30), 'hwi': (35, 55)},
            'Swift': {'wl': (150, 220), 's1': (70, 120), 'kd': (80, 140), 'hwi': (45, 70)},
            'Swallow': {'wl': (100, 160), 's1': (60, 100), 'kd': (40, 80), 'hwi': (35, 60)},
            'Hawk': {'wl': (300, 500), 's1': (180, 300), 'kd': (120, 250), 'hwi': (20, 45)},
            'Eagle': {'wl': (500, 800), 's1': (300, 500), 'kd': (200, 400), 'hwi': (15, 40)},
            'Albatross': {'wl': (600, 900), 's1': (250, 400), 'kd': (350, 600), 'hwi': (50, 80)},
            'Falcon': {'wl': (250, 400), 's1': (140, 240), 'kd': (110, 200), 'hwi': (55, 85)},
            'Crow': {'wl': (250, 350), 's1': (170, 250), 'kd': (80, 150), 'hwi': (25, 50)}
        }
        
        bird_list = []
        for archetype, ranges in archetypes.items():
            for i in range(20):  # 20 variants per archetype
                variation = np.random.normal(1.0, 0.2)  # 20% variation
                bird = {
                    'species': f"{archetype}_{i+1:02d}",
                    'Wing.Length': max(20, ranges['wl'][0] + (ranges['wl'][1] - ranges['wl'][0]) * np.random.random()) * variation,
                    'Secondary1': max(10, ranges['s1'][0] + (ranges['s1'][1] - ranges['s1'][0]) * np.random.random()) * variation,
                    'Kipps.Distance': max(5, ranges['kd'][0] + (ranges['kd'][1] - ranges['kd'][0]) * np.random.random()) * variation,
                    'Hand-Wing.Index': max(10, ranges['hwi'][0] + (ranges['hwi'][1] - ranges['hwi'][0]) * np.random.random()) * variation
                }
                bird_list.append(bird)
        
        return pd.DataFrame(bird_list)
    
    def create_environments(self):
        """Create training and evaluation environments"""
        def make_env():
            env = AdvancedAirfoilEnvironment(
                bird_data=self.bird_data,
                use_biological_constraints=self.config.use_biological_constraints,
                multi_objective=self.config.multi_objective,
                adaptive_targets=self.config.adaptive_targets,
                reynolds_number=self.config.reynolds_number
            )
            return Monitor(env)
        
        # Create vectorized training environment
        if self.config.n_parallel_envs > 1:
            self.train_env = SubprocVecEnv([make_env for _ in range(self.config.n_parallel_envs)])
        else:
            self.train_env = DummyVecEnv([make_env])
        
        # Create evaluation environment
        self.eval_env = make_env()
        
        logger.info(f"Created training environment with {self.config.n_parallel_envs} parallel processes")
    
    def create_model(self):
        """Create RL model based on configuration"""
        logger.info(f"Creating {self.config.algorithm} model")
        
        if self.config.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                self.train_env,
                learning_rate=self.config.learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                normalize_advantage=True,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log=self.config.log_dir,
                verbose=1,
                device='auto'
            )
        elif self.config.algorithm == 'SAC':
            self.model = SAC(
                'MlpPolicy',
                self.train_env,
                learning_rate=self.config.learning_rate,
                buffer_size=300000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef='auto',
                target_update_interval=1,
                tensorboard_log=self.config.log_dir,
                verbose=1,
                device='auto'
            )
        elif self.config.algorithm == 'TD3':
            self.model = TD3(
                'MlpPolicy',
                self.train_env,
                learning_rate=self.config.learning_rate,
                buffer_