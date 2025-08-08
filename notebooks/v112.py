import numpy as np
import gym
from gym import spaces
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.special import comb
from scipy.optimize import minimize
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class AirfoilMetrics:
    """Comprehensive aerodynamic performance metrics"""
    cl: float  # Lift coefficient
    cd: float  # Drag coefficient
    cm: float  # Moment coefficient
    cl_cd_ratio: float  # Lift-to-drag ratio
    stall_angle: float  # Stall angle of attack
    max_camber: float  # Maximum camber
    max_thickness: float  # Maximum thickness
    camber_position: float  # Position of maximum camber
    thickness_position: float  # Position of maximum thickness
    leading_edge_radius: float  # Leading edge radius
    trailing_edge_angle: float  # Trailing edge angle
    reynolds_number: float  # Reynolds number
    mach_number: float  # Mach number


class BiologicalConstraints:
    """Enforces biological constraints on airfoil shapes"""
    
    def __init__(self):
        # Biological ranges from bird wing analysis
        self.thickness_range = (0.03, 0.25)  # 3-25% thickness-to-chord ratio
        self.camber_range = (0.0, 0.12)     # 0-12% camber
        self.aspect_ratio_range = (5.0, 20.0)  # Wing aspect ratio
        self.taper_ratio_range = (0.3, 1.0)    # Wing taper
        
    def apply_thickness_constraint(self, thickness_dist: np.ndarray) -> np.ndarray:
        """Apply biological thickness constraints"""
        max_thickness = np.max(thickness_dist)
        if max_thickness < self.thickness_range[0]:
            thickness_dist *= self.thickness_range[0] / max_thickness
        elif max_thickness > self.thickness_range[1]:
            thickness_dist *= self.thickness_range[1] / max_thickness
        return thickness_dist
    
    def apply_camber_constraint(self, camber_line: np.ndarray) -> np.ndarray:
        """Apply biological camber constraints"""
        max_camber = np.max(np.abs(camber_line))
        if max_camber > self.camber_range[1]:
            camber_line *= self.camber_range[1] / max_camber
        return camber_line
    
    def check_biological_plausibility(self, airfoil_coords: np.ndarray) -> float:
        """Return biological plausibility score (0-1)"""
        upper_curve, lower_curve = airfoil_coords
        
        # Calculate geometric properties
        camber_line = (upper_curve[:, 1] + lower_curve[:, 1]) / 2
        thickness_dist = upper_curve[:, 1] - lower_curve[:, 1]
        
        max_camber = np.max(camber_line)
        max_thickness = np.max(thickness_dist)
        
        # Score based on biological ranges
        thickness_score = 1.0 if self.thickness_range[0] <= max_thickness <= self.thickness_range[1] else 0.5
        camber_score = 1.0 if max_camber <= self.camber_range[1] else 0.5
        
        # Smoothness penalty
        upper_curvature = np.gradient(np.gradient(upper_curve[:, 1]))
        lower_curvature = np.gradient(np.gradient(lower_curve[:, 1]))
        smoothness_score = 1.0 / (1.0 + np.mean(np.abs(upper_curvature)) + np.mean(np.abs(lower_curvature)))
        
        return (thickness_score + camber_score + smoothness_score) / 3.0


class AdvancedAirfoilEnvironment(gym.Env):
    """
    Advanced RL environment for bio-inspired airfoil optimization
    Integrates with XFOIL simulation and biological constraints
    """
    
    def __init__(self, 
                 bird_data: pd.DataFrame,
                 target_performance: Dict = None,
                 use_biological_constraints: bool = True,
                 reynolds_number: float = 1e6,
                 mach_number: float = 0.1,
                 multi_objective: bool = True,
                 adaptive_targets: bool = True):
        super().__init__()
        
        self.bird_data = bird_data.reset_index(drop=True)
        self.reynolds = reynolds_number
        self.mach = mach_number
        self.multi_objective = multi_objective
        self.adaptive_targets = adaptive_targets
        
        # Target performance metrics
        self.target_performance = target_performance or {
            'cl_cd_ratio': 100.0,
            'cl_max': 1.4,
            'cd_min': 0.008,
            'stall_angle': 14.0
        }
        
        # Biological constraints
        self.bio_constraints = BiologicalConstraints() if use_biological_constraints else None
        
        # Enhanced action space: 
        # - 20 control points (10 upper, 10 lower) with y-coordinates
        # - 8 global parameters (thickness, camber, twist, etc.)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(28,),  # 20 + 8
            dtype=np.float32
        )
        
        # Enhanced observation space:
        # - Current airfoil shape (40 points)
        # - Performance metrics (12 values)
        # - Target metrics (8 values)
        # - Bird morphology features (10 values)
        # - Training progress (5 values)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(75,),
            dtype=np.float32
        )
        
        # Initialize environment state
        self.current_bird_idx = 0
        self.current_airfoil = None
        self.episode_history = []
        self.performance_history = []
        self.convergence_threshold = 1e-4
        
        # Adaptive curriculum learning
        self.difficulty_level = 0.1  # Start easy
        self.success_rate_window = []
        
        self.reset()
    
    def reset(self):
        """Reset environment to initial state"""
        # Select target bird (curriculum learning)
        if self.adaptive_targets:
            self.current_bird_idx = self._select_adaptive_target()
        else:
            self.current_bird_idx = np.random.randint(len(self.bird_data))
        
        self.current_bird = self.bird_data.iloc[self.current_bird_idx]
        
        # Initialize airfoil with biological prior
        self.current_airfoil = self._initialize_biological_airfoil()
        
        # Reset episode tracking
        self.episode_step = 0
        self.episode_history = []
        self.best_performance = -np.inf
        
        # Calculate adaptive targets based on bird morphology
        if self.adaptive_targets:
            self.target_performance = self._calculate_adaptive_targets()
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one environment step"""
        self.episode_step += 1
        
        # Apply action to modify airfoil
        self._apply_action(action)
        
        # Evaluate airfoil performance
        performance_metrics = self._evaluate_performance()
        
        # Calculate multi-objective reward
        reward = self._calculate_reward(performance_metrics)
        
        # Check termination conditions
        done = self._check_termination(performance_metrics)
        
        # Update episode history
        self.episode_history.append({
            'step': self.episode_step,
            'action': action.copy(),
            'performance': performance_metrics,
            'reward': reward
        })
        
        # Adaptive curriculum update
        if done and self.adaptive_targets:
            self._update_curriculum(reward)
        
        # Comprehensive info dictionary
        info = {
            'performance_metrics': performance_metrics,
            'target_performance': self.target_performance,
            'bird_species': self.current_bird.get('species', 'unknown'),
            'difficulty_level': self.difficulty_level,
            'biological_plausibility': self._calculate_biological_plausibility(),
            'convergence_rate': self._calculate_convergence_rate()
        }
        
        return self._get_observation(), reward, done, info
    
    def _select_adaptive_target(self):
        """Select target bird based on current difficulty level"""
        # Sort birds by complexity (wing aspect ratio, hand-wing index)
        if 'complexity_score' not in self.bird_data.columns:
            self._calculate_bird_complexity()
        
        # Select based on difficulty level
        n_birds = len(self.bird_data)
        difficulty_idx = int(self.difficulty_level * (n_birds - 1))
        
        # Add some randomness
        difficulty_range = max(1, int(0.1 * n_birds))
        start_idx = max(0, difficulty_idx - difficulty_range)
        end_idx = min(n_birds, difficulty_idx + difficulty_range)
        
        return np.random.randint(start_idx, end_idx)
    
    def _calculate_bird_complexity(self):
        """Calculate complexity score for each bird"""
        # Normalize morphological features
        features = ['Wing.Length', 'Secondary1', 'Kipps.Distance', 'Hand-Wing.Index']
        
        complexity_scores = []
        for _, bird in self.bird_data.iterrows():
            # Higher aspect ratio = more complex
            aspect_ratio = bird['Wing.Length'] / bird['Secondary1'] if bird['Secondary1'] > 0 else 5.0
            
            # Higher hand-wing index = more specialized/complex
            hand_wing_complexity = bird['Hand-Wing.Index'] / 100.0
            
            # Wing tip complexity
            kipps_ratio = bird['Kipps.Distance'] / bird['Wing.Length'] if bird['Wing.Length'] > 0 else 0.3
            
            # Combined complexity
            complexity = (aspect_ratio / 20.0) + hand_wing_complexity + kipps_ratio
            complexity_scores.append(complexity)
        
        self.bird_data['complexity_score'] = complexity_scores
    
    def _initialize_biological_airfoil(self):
        """Initialize airfoil based on current bird morphology"""
        bird = self.current_bird
        
        # Extract morphological ratios
        wing_length = bird['Wing.Length']
        secondary1 = bird['Secondary1']
        kipps_distance = bird['Kipps.Distance']
        hand_wing_index = bird['Hand-Wing.Index']
        
        # Calculate ratios
        secondary_ratio = secondary1 / wing_length if wing_length > 0 else 0.6
        kipps_ratio = kipps_distance / wing_length if wing_length > 0 else 0.3
        hand_wing_ratio = hand_wing_index / 100.0
        
        # Biological airfoil parameters
        max_thickness = 0.08 + secondary_ratio * 0.08 - hand_wing_ratio * 0.04
        max_thickness = np.clip(max_thickness, 0.04, 0.20)
        
        camber = 0.02 + kipps_ratio * 0.06 + hand_wing_ratio * 0.04
        camber = np.clip(camber, 0.0, 0.10)
        
        # Generate control points
        n_points = 10
        x_coords = np.linspace(0, 1, n_points)
        
        # Upper surface (using biological shape)
        upper_y = np.zeros(n_points)
        for i, x in enumerate(x_coords):
            if x <= 0.3:  # Leading edge region
                upper_y[i] = camber + max_thickness * np.sqrt(x / 0.3)
            else:  # Trailing edge region
                upper_y[i] = camber + max_thickness * (1 - x)**1.5
        
        # Lower surface
        lower_y = np.zeros(n_points)
        for i, x in enumerate(x_coords):
            if x <= 0.3:
                lower_y[i] = camber - max_thickness * 0.6 * np.sqrt(x / 0.3)
            else:
                lower_y[i] = camber - max_thickness * 0.4 * (1 - x)**1.5
        
        # Ensure trailing edge closure
        upper_y[-1] = 0
        lower_y[-1] = 0
        
        return {
            'upper_cp': np.column_stack([x_coords, upper_y]),
            'lower_cp': np.column_stack([x_coords, lower_y]),
            'parameters': {
                'thickness_scale': 1.0,
                'camber_scale': 1.0,
                'twist': 0.0,
                'aspect_ratio': 10.0,
                'taper_ratio': 0.6,
                'sweep_angle': 0.0,
                'dihedral_angle': 0.0,
                'reynolds_correction': 1.0
            }
        }
    
    def _apply_action(self, action):
        """Apply RL action to modify airfoil"""
        # Split action into control points and parameters
        cp_actions = action[:20] * 0.01  # Small modifications
        param_actions = action[20:] * 0.1  # Parameter modifications
        
        # Modify control points
        upper_cp = self.current_airfoil['upper_cp'].copy()
        lower_cp = self.current_airfoil['lower_cp'].copy()
        
        # Update y-coordinates (preserve x-coordinates for stability)
        for i in range(1, len(upper_cp) - 1):  # Skip endpoints
            upper_cp[i, 1] += cp_actions[i - 1]
            lower_cp[i, 1] += cp_actions[i - 1 + 9]
        
        # Apply biological constraints
        if self.bio_constraints:
            thickness_dist = upper_cp[:, 1] - lower_cp[:, 1]
            thickness_dist = self.bio_constraints.apply_thickness_constraint(thickness_dist)
            
            # Redistribute thickness
            camber_line = (upper_cp[:, 1] + lower_cp[:, 1]) / 2
            upper_cp[:, 1] = camber_line + thickness_dist / 2
            lower_cp[:, 1] = camber_line - thickness_dist / 2
        
        # Update global parameters
        params = self.current_airfoil['parameters']
        param_names = list(params.keys())
        for i, name in enumerate(param_names[:len(param_actions)]):
            if name == 'thickness_scale':
                params[name] = np.clip(params[name] + param_actions[i], 0.5, 2.0)
            elif name == 'camber_scale':
                params[name] = np.clip(params[name] + param_actions[i], 0.1, 3.0)
            elif name == 'twist':
                params[name] = np.clip(params[name] + param_actions[i], -10.0, 10.0)
            elif name == 'aspect_ratio':
                params[name] = np.clip(params[name] + param_actions[i], 5.0, 25.0)
            else:
                params[name] += param_actions[i]
        
        # Update airfoil
        self.current_airfoil.update({
            'upper_cp': upper_cp,
            'lower_cp': lower_cp,
            'parameters': params
        })
    
    def _evaluate_performance(self):
        """Evaluate airfoil aerodynamic performance"""
        # Generate smooth curves from control points
        upper_curve = self._generate_bezier_curve(self.current_airfoil['upper_cp'])
        lower_curve = self._generate_bezier_curve(self.current_airfoil['lower_cp'])
        
        # Apply global parameters
        upper_curve, lower_curve = self._apply_global_parameters(upper_curve, lower_curve)
        
        # Calculate aerodynamic properties (simplified model)
        # In production, this would interface with XFOIL or CFD
        metrics = self._calculate_aerodynamic_properties(upper_curve, lower_curve)
        
        return metrics
    
    def _generate_bezier_curve(self, control_points, num_points=100):
        """Generate Bézier curve from control points"""
        n = len(control_points) - 1
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))
        
        for i in range(num_points):
            for j in range(n + 1):
                bernstein = comb(n, j) * (1 - t[i])**(n - j) * t[i]**j
                curve[i] += bernstein * control_points[j]
        
        return curve
    
    def _apply_global_parameters(self, upper_curve, lower_curve):
        """Apply global shape parameters"""
        params = self.current_airfoil['parameters']
        
        # Apply scaling
        thickness_scale = params['thickness_scale']
        camber_scale = params['camber_scale']
        
        # Calculate current thickness and camber
        thickness = (upper_curve[:, 1] - lower_curve[:, 1]) * thickness_scale
        camber_line = (upper_curve[:, 1] + lower_curve[:, 1]) / 2 * camber_scale
        
        # Apply modifications
        upper_modified = upper_curve.copy()
        lower_modified = lower_curve.copy()
        
        upper_modified[:, 1] = camber_line + thickness / 2
        lower_modified[:, 1] = camber_line - thickness / 2
        
        # Apply twist (rotation about quarter chord)
        twist_rad = np.radians(params['twist'])
        if abs(twist_rad) > 1e-6:
            quarter_chord = 0.25
            for curve in [upper_modified, lower_modified]:
                for i in range(len(curve)):
                    x_rel = curve[i, 0] - quarter_chord
                    y_rel = curve[i, 1]
                    curve[i, 0] = quarter_chord + x_rel * np.cos(twist_rad) - y_rel * np.sin(twist_rad)
                    curve[i, 1] = x_rel * np.sin(twist_rad) + y_rel * np.cos(twist_rad)
        
        return upper_modified, lower_modified
    
    def _calculate_aerodynamic_properties(self, upper_curve, lower_curve):
        """Calculate aerodynamic properties using empirical methods"""
        # Geometric analysis
        x_coords = upper_curve[:, 0]
        camber_line = (upper_curve[:, 1] + lower_curve[:, 1]) / 2
        thickness_dist = upper_curve[:, 1] - lower_curve[:, 1]
        
        # Key parameters
        max_camber = np.max(camber_line)
        max_thickness = np.max(thickness_dist)
        max_camber_pos = x_coords[np.argmax(camber_line)]
        max_thickness_pos = x_coords[np.argmax(thickness_dist)]
        
        # Leading edge radius approximation
        if len(upper_curve) > 2:
            dx1 = upper_curve[1, 0] - upper_curve[0, 0]
            dy1 = upper_curve[1, 1] - upper_curve[0, 1]
            dx2 = upper_curve[2, 0] - upper_curve[1, 0]
            dy2 = upper_curve[2, 1] - upper_curve[1, 1]
            
            curvature = abs(dx1 * dy2 - dx2 * dy1) / ((dx1**2 + dy1**2)**1.5 + 1e-8)
            leading_edge_radius = 1.0 / (curvature + 1e-8)
            leading_edge_radius = min(leading_edge_radius, 0.1)
        else:
            leading_edge_radius = 0.01
        
        # Trailing edge angle
        if len(upper_curve) > 2:
            upper_slope = (upper_curve[-1, 1] - upper_curve[-3, 1]) / (upper_curve[-1, 0] - upper_curve[-3, 0] + 1e-8)
            lower_slope = (lower_curve[-1, 1] - lower_curve[-3, 1]) / (lower_curve[-1, 0] - lower_curve[-3, 0] + 1e-8)
            trailing_edge_angle = abs(np.arctan(upper_slope) - np.arctan(lower_slope))
        else:
            trailing_edge_angle = 0.1
        
        # Advanced empirical correlations
        # Lift coefficient slope (per radian)
        a0 = 2 * np.pi * (1 + 0.77 * max_thickness)
        
        # Maximum lift coefficient
        cl_max = 0.9 + 4.0 * max_camber + 1.2 * max_thickness - 2.0 * max_camber**2
        cl_max = np.clip(cl_max, 0.5, 2.5)
        
        # Profile drag coefficient
        cd_profile = 0.006 + 0.02 * max_thickness**2 + 0.1 * max_camber**2
        
        # Reynolds number effects
        reynolds_factor = (self.reynolds / 1e6) ** (-0.2)
        cd_profile *= reynolds_factor
        
        # Pressure drag from thickness distribution
        thickness_penalty = np.mean(np.gradient(thickness_dist)**2) * 0.001
        cd_profile += thickness_penalty
        
        # Moment coefficient
        cm_quarter = -0.05 - 0.1 * max_camber * (max_camber_pos - 0.25)
        
        # Stall characteristics
        stall_angle = 16 - 20 * max_thickness + 10 * max_camber
        stall_angle = np.clip(stall_angle, 8, 20)
        
        # Critical angle of attack for optimal L/D
        optimal_alpha = 4.0 + max_camber * 30
        optimal_cl = 0.5 + 8 * max_camber
        
        # Calculate L/D ratio
        cd_min = cd_profile
        cl_cd_ratio = optimal_cl / cd_min if cd_min > 1e-6 else 100.0
        
        return AirfoilMetrics(
            cl=optimal_cl,
            cd=cd_min,
            cm=cm_quarter,
            cl_cd_ratio=cl_cd_ratio,
            stall_angle=stall_angle,
            max_camber=max_camber,
            max_thickness=max_thickness,
            camber_position=max_camber_pos,
            thickness_position=max_thickness_pos,
            leading_edge_radius=leading_edge_radius,
            trailing_edge_angle=np.degrees(trailing_edge_angle),
            reynolds_number=self.reynolds,
            mach_number=self.mach
        )
    
    def _calculate_reward(self, metrics: AirfoilMetrics):
        """Calculate multi-objective reward"""
        if not self.multi_objective:
            # Single objective: maximize L/D ratio
            target_ld = self.target_performance['cl_cd_ratio']
            error = abs(metrics.cl_cd_ratio - target_ld) / target_ld
            return np.exp(-error)
        
        # Multi-objective optimization
        objectives = {}
        
        # L/D ratio objective
        ld_target = self.target_performance['cl_cd_ratio']
        ld_error = abs(metrics.cl_cd_ratio - ld_target) / ld_target
        objectives['lift_drag'] = np.exp(-ld_error) * 0.4
        
        # Lift coefficient objective
        cl_target = self.target_performance['cl_max']
        cl_error = abs(metrics.cl - cl_target) / cl_target
        objectives['lift'] = np.exp(-cl_error) * 0.3
        
        # Drag minimization objective
        cd_target = self.target_performance['cd_min']
        cd_error = abs(metrics.cd - cd_target) / cd_target
        objectives['drag'] = np.exp(-cd_error) * 0.2
        
        # Stall characteristics
        stall_target = self.target_performance['stall_angle']
        stall_error = abs(metrics.stall_angle - stall_target) / stall_target
        objectives['stall'] = np.exp(-stall_error) * 0.1
        
        # Biological plausibility bonus
        if self.bio_constraints:
            bio_score = self._calculate_biological_plausibility()
            objectives['biological'] = bio_score * 0.15
        
        # Smoothness penalty
        smoothness_penalty = self._calculate_smoothness_penalty()
        objectives['smoothness'] = (1.0 - smoothness_penalty) * 0.1
        
        total_reward = sum(objectives.values())
        
        # Progressive reward (encourage improvement)
        if hasattr(self, 'previous_reward'):
            improvement = total_reward - self.previous_reward
            total_reward += improvement * 0.05
        
        self.previous_reward = total_reward
        
        return float(total_reward)
    
    def _calculate_biological_plausibility(self):
        """Calculate biological plausibility score"""
        if not self.bio_constraints:
            return 1.0
        
        upper_curve = self._generate_bezier_curve(self.current_airfoil['upper_cp'])
        lower_curve = self._generate_bezier_curve(self.current_airfoil['lower_cp'])
        
        return self.bio_constraints.check_biological_plausibility((upper_curve, lower_curve))
    
    def _calculate_smoothness_penalty(self):
        """Calculate smoothness penalty based on curvature discontinuities"""
        upper_cp = self.current_airfoil['upper_cp']
        lower_cp = self.current_airfoil['lower_cp']
        
        # Calculate second derivatives (curvature)
        upper_curvature = 0
        lower_curvature = 0
        
        for i in range(1, len(upper_cp) - 1):
            d2y = upper_cp[i+1, 1] - 2*upper_cp[i, 1] + upper_cp[i-1, 1]
            upper_curvature += abs(d2y)
        
        for i in range(1, len(lower_cp) - 1):
            d2y = lower_cp[i+1, 1] - 2*lower_cp[i, 1] + lower_cp[i-1, 1]
            lower_curvature += abs(d2y)
        
        total_curvature = (upper_curvature + lower_curvature) / (len(upper_cp) + len(lower_cp))
        return min(total_curvature, 1.0)
    
    def _calculate_adaptive_targets(self):
        """Calculate performance targets based on bird morphology"""
        bird = self.current_bird
        
        # Extract morphological features
        wing_length = bird['Wing.Length']
        secondary1 = bird['Secondary1']
        kipps_distance = bird['Kipps.Distance']
        hand_wing_index = bird['Hand-Wing.Index']
        
        # Calculate ratios
        aspect_ratio_est = wing_length / secondary1 if secondary1 > 0 else 8.0
        kipps_ratio = kipps_distance / wing_length if wing_length > 0 else 0.3
        hand_wing_ratio = hand_wing_index / 100.0
        
        # Adaptive targets based on flight regime
        if hand_wing_ratio > 0.4 and kipps_ratio > 0.4:
            # High-speed specialists
            target_ld = 80 + hand_wing_ratio * 50
            target_cl = 1.0 + kipps_ratio * 0.6
            target_cd = 0.008 + hand_wing_ratio * 0.004
            target_stall = 12 + kipps_ratio * 4
        elif aspect_ratio_est > 10:
            # Soaring specialists
            target_ld = 120 + aspect_ratio_est * 3
            target_cl = 1.2 + kipps_ratio * 0.4
            target_cd = 0.006 + hand_wing_ratio * 0.002
            target_stall = 14 + kipps_ratio * 2
        else:
            # Generalists
            target_ld = 70 + hand_wing_ratio * 20
            target_cl = 1.3 + kipps_ratio * 0.3
            target_cd = 0.010 + hand_wing_ratio * 0.003
            target_stall = 15 + kipps_ratio * 3
        
        return {
            'cl_cd_ratio': float(target_ld),
            'cl_max': float(target_cl),
            'cd_min': float(target_cd),
            'stall_angle': float(target_stall)
        }
    
    def _check_termination(self, metrics: AirfoilMetrics):
        """Check if episode should terminate"""
        # Maximum episode length
        if self.episode_step >= 200:
            return True
        
        # Performance target achieved
        targets = self.target_performance
        tolerance = 0.05  # 5% tolerance
        
        ld_achieved = abs(metrics.cl_cd_ratio - targets['cl_cd_ratio']) / targets['cl_cd_ratio'] < tolerance
        cl_achieved = abs(metrics.cl - targets['cl_max']) / targets['cl_max'] < tolerance
        
        if ld_achieved and cl_achieved:
            return True
        
        # Convergence check
        if len(self.episode_history) > 10:
            recent_rewards = [h['reward'] for h in self.episode_history[-10:]]
            if np.std(recent_rewards) < self.convergence_threshold:
                return True
        
        return False
    
    def _update_curriculum(self, episode_reward):
        """Update curriculum difficulty based on performance"""
        self.success_rate_window.append(episode_reward > 0.8)  # Success threshold
        
        # Keep window size manageable
        if len(self.success_rate_window) > 50:
            self.success_rate_window.pop(0)
        
        # Adjust difficulty based on success rate
        if len(self.success_rate_window) >= 10:
            success_rate = np.mean(self.success_rate_window)
            
            if success_rate > 0.8:  # Too easy, increase difficulty
                self.difficulty_level = min(1.0, self.difficulty_level + 0.05)
            elif success_rate < 0.3:  # Too hard, decrease difficulty
                self.difficulty_level = max(0.0, self.difficulty_level - 0.03)
    
    def _calculate_convergence_rate(self):
        """Calculate convergence rate metric"""
        if len(self.episode_history) < 5:
            return 0.0
        
        recent_rewards = [h['reward'] for h in self.episode_history[-5:]]
        return float(np.std(recent_rewards))
    
    def _get_observation(self):
        """Get current observation state"""
        # Current airfoil shape (upper + lower control points y-coordinates)
        upper_y = self.current_airfoil['upper_cp'][:, 1]
        lower_y = self.current_airfoil['lower_cp'][:, 1]
        shape_obs = np.concatenate([upper_y, lower_y])  # 20 values
        
        # Pad/truncate to ensure consistent size
        shape_obs = np.pad(shape_obs, (0, max(0, 20 - len(shape_obs))))[:20]
        
        # Global parameters
        params = self.current_airfoil['parameters']
        param_obs = np.array([
            params['thickness_scale'],
            params['camber_scale'],
            params['twist'],
            params['aspect_ratio'] / 20.0,  # Normalize
            params['taper_ratio'],
            params['sweep_angle'] / 30.0,   # Normalize
            params['dihedral_angle'] / 10.0,  # Normalize
            params['reynolds_correction']
        ])
        
        # Current performance (if available)
        if hasattr(self, 'last_metrics'):
            perf_obs = np.array([
                self.last_metrics.cl,
                self.last_metrics.cd * 100,  # Scale up
                self.last_metrics.cl_cd_ratio / 100,  # Scale down
                self.last_metrics.stall_angle / 20.0,  # Normalize
                self.last_metrics.max_camber,
                self.last_metrics.max_thickness,
                self.last_metrics.camber_position,
                self.last_metrics.thickness_position,
                self.last_metrics.leading_edge_radius * 100,  # Scale up
                self.last_metrics.trailing_edge_angle / 30.0,  # Normalize
                self.reynolds / 1e6,  # Scale down
                self.mach
            ])
        else:
            perf_obs = np.zeros(12)
        
        # Target performance
        target_obs = np.array([
            self.target_performance['cl_cd_ratio'] / 100,
            self.target_performance['cl_max'],
            self.target_performance['cd_min'] * 100,
            self.target_performance['stall_angle'] / 20.0
        ])
        
        # Bird morphology features
        bird = self.current_bird
        morph_obs = np.array([
            bird['Wing.Length'] / 1000.0,  # Normalize to [0,1] range
            bird['Secondary1'] / 1000.0,
            bird['Kipps.Distance'] / 1000.0,
            bird['Hand-Wing.Index'] / 100.0,
            bird.get('Tail.Length', 100) / 1000.0,
            bird['Secondary1'] / bird['Wing.Length'] if bird['Wing.Length'] > 0 else 0.6,  # Secondary ratio
            bird['Kipps.Distance'] / bird['Wing.Length'] if bird['Wing.Length'] > 0 else 0.3,  # Kipps ratio
            bird['Wing.Length'] / bird['Secondary1'] if bird['Secondary1'] > 0 else 8.0,  # Aspect ratio est
            self.difficulty_level,
            float(self.current_bird_idx) / len(self.bird_data)
        ])
        
        # Training progress
        progress_obs = np.array([
            self.episode_step / 200.0,  # Normalize by max episode length
            len(self.episode_history) / 200.0,
            self.difficulty_level,
            float(len(self.success_rate_window)) / 50.0 if hasattr(self, 'success_rate_window') else 0.0,
            np.mean(self.success_rate_window) if hasattr(self, 'success_rate_window') and len(self.success_rate_window) > 0 else 0.0
        ])
        
        # Combine all observations
        obs = np.concatenate([
            shape_obs[:20],      # 20 values
            param_obs[:8],       # 8 values  
            perf_obs[:12],       # 12 values
            target_obs[:4],      # 4 values
            morph_obs[:10],      # 10 values
            progress_obs[:5]     # 5 values
        ])  # Total: 59 values
        
        # Pad to match observation space
        obs = np.pad(obs, (0, max(0, 75 - len(obs))))[:75]
        
        return obs.astype(np.float32)
    
    def render(self, mode='human'):
        """Render current airfoil"""
        if mode == 'human':
            plt.figure(figsize=(12, 8))
            
            # Generate curves
            upper_curve = self._generate_bezier_curve(self.current_airfoil['upper_cp'])
            lower_curve = self._generate_bezier_curve(self.current_airfoil['lower_cp'])
            upper_curve, lower_curve = self._apply_global_parameters(upper_curve, lower_curve)
            
            # Plot airfoil
            plt.subplot(2, 2, 1)
            plt.plot(upper_curve[:, 0], upper_curve[:, 1], 'b-', linewidth=2, label='Upper')
            plt.plot(lower_curve[:, 0], lower_curve[:, 1], 'r-', linewidth=2, label='Lower')
            plt.fill_between(upper_curve[:, 0], upper_curve[:, 1], lower_curve[:, 1], 
                           alpha=0.3, color='lightblue')
            
            # Plot control points
            plt.plot(self.current_airfoil['upper_cp'][:, 0], 
                    self.current_airfoil['upper_cp'][:, 1], 'bo--', alpha=0.5)
            plt.plot(self.current_airfoil['lower_cp'][:, 0], 
                    self.current_airfoil['lower_cp'][:, 1], 'ro--', alpha=0.5)
            
            plt.xlim(-0.1, 1.1)
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.title(f'Current Airfoil - Step {self.episode_step}')
            plt.legend()
            
            # Performance metrics
            if hasattr(self, 'last_metrics'):
                plt.subplot(2, 2, 2)
                metrics = ['L/D', 'CL', 'CD×100', 'Stall°']
                values = [self.last_metrics.cl_cd_ratio, self.last_metrics.cl, 
                         self.last_metrics.cd*100, self.last_metrics.stall_angle]
                targets = [self.target_performance['cl_cd_ratio'], 
                          self.target_performance['cl_max'],
                          self.target_performance['cd_min']*100,
                          self.target_performance['stall_angle']]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                plt.bar(x - width/2, values, width, label='Current', alpha=0.7)
                plt.bar(x + width/2, targets, width, label='Target', alpha=0.7)
                plt.xticks(x, metrics)
                plt.title('Performance Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Episode history
            if self.episode_history:
                plt.subplot(2, 2, 3)
                rewards = [h['reward'] for h in self.episode_history]
                plt.plot(rewards, 'g-', linewidth=2)
                plt.title('Episode Reward History')
                plt.xlabel('Step')
                plt.ylabel('Reward')
                plt.grid(True, alpha=0.3)
            
            # Bird information
            plt.subplot(2, 2, 4)
            bird_info = [
                f"Species: {self.current_bird.get('species', 'Unknown')[:20]}",
                f"Wing Length: {self.current_bird['Wing.Length']:.1f} mm",
                f"Secondary1: {self.current_bird['Secondary1']:.1f} mm", 
                f"Kipps Distance: {self.current_bird['Kipps.Distance']:.1f} mm",
                f"Hand-Wing Index: {self.current_bird['Hand-Wing.Index']:.1f}",
                f"Difficulty: {self.difficulty_level:.2f}",
                f"Episode: {self.episode_step}/200"
            ]
            
            for i, info in enumerate(bird_info):
                plt.text(0.05, 0.9 - i*0.12, info, transform=plt.gca().transAxes, 
                        fontsize=10, verticalalignment='top')
            
            plt.axis('off')
            plt.title('Target Bird Information')
            
            plt.tight_layout()
            plt.show()
    
    def get_airfoil_coordinates(self, num_points=200):
        """Get current airfoil coordinates for export"""
        upper_curve = self._generate_bezier_curve(self.current_airfoil['upper_cp'], num_points)
        lower_curve = self._generate_bezier_curve(self.current_airfoil['lower_cp'], num_points)
        upper_curve, lower_curve = self._apply_global_parameters(upper_curve, lower_curve)
        
        # Combine for XFOIL format (upper surface + reversed lower surface)
        coords = np.vstack([upper_curve, lower_curve[::-1]])
        return coords
    
    def save_airfoil(self, filename):
        """Save current airfoil to file"""
        coords = self.get_airfoil_coordinates()
        
        header = f"""# Bio-inspired airfoil generated by RL
# Target bird: {self.current_bird.get('species', 'Unknown')}
# Wing Length: {self.current_bird['Wing.Length']:.1f} mm
# Performance: L/D = {getattr(self, 'last_metrics', AirfoilMetrics(0,0,0,0,0,0,0,0,0,0,0,0,0)).cl_cd_ratio:.1f}
# Episode step: {self.episode_step}
# Coordinates: x y (normalized chord)"""
        
        np.savetxt(filename, coords, header=header, fmt='%.6f', delimiter=' ')
        logger.info(f"Airfoil saved to {filename}")
    
    def export_episode_data(self):
        """Export detailed episode data for analysis"""
        return {
            'bird_data': self.current_bird.to_dict(),
            'target_performance': self.target_performance,
            'episode_history': self.episode_history,
            'final_airfoil': {
                'upper_cp': self.current_airfoil['upper_cp'].tolist(),
                'lower_cp': self.current_airfoil['lower_cp'].tolist(),
                'parameters': self.current_airfoil['parameters']
            },
            'difficulty_level': self.difficulty_level,
            'biological_plausibility': self._calculate_biological_plausibility()
        }