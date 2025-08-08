import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import comb
from scipy.optimize import minimize
import gym
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import subprocess
import os
import time
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AirfoilMetrics:
    """Structured container for airfoil performance metrics"""
    cl: float  # Lift coefficient
    cd: float  # Drag coefficient
    cm: float  # Moment coefficient
    cl_cd_ratio: float  # Lift-to-drag ratio
    stall_angle: float  # Stall angle
    max_camber: float  # Maximum camber
    max_thickness: float  # Maximum thickness
    reynolds_number: float  # Reynolds number

class XFOILSimulator:
    """Advanced XFOIL integration for accurate airfoil analysis"""
    
    def __init__(self, xfoil_path: str = "xfoil", work_dir: str = "./xfoil_work"):
        self.xfoil_path = xfoil_path
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        
    def analyze_airfoil(self, upper_curve: np.ndarray, lower_curve: np.ndarray, 
                       reynolds: float = 1e6, mach: float = 0.0,
                       alpha_range: Tuple[float, float] = (-5, 15)) -> Dict:
        """
        Perform comprehensive XFOIL analysis of airfoil
        
        Args:
            upper_curve: Upper surface coordinates
            lower_curve: Lower surface coordinates  
            reynolds: Reynolds number
            mach: Mach number
            alpha_range: Range of angles of attack to analyze
            
        Returns:
            Dictionary with comprehensive aerodynamic data
        """
        
        # Generate airfoil coordinates file
        airfoil_file = os.path.join(self.work_dir, "airfoil.dat")
        self._write_airfoil_coordinates(upper_curve, lower_curve, airfoil_file)
        
        # Create XFOIL input script
        script_file = os.path.join(self.work_dir, "xfoil_script.txt")
        output_file = os.path.join(self.work_dir, "polar.dat")
        
        xfoil_commands = f"""
LOAD {airfoil_file}

PANEL

OPER
VISC {reynolds}
MACH {mach}

PACC
{output_file}


ASEQ {alpha_range[0]} {alpha_range[1]} 0.5

QUIT
"""
        
        with open(script_file, 'w') as f:
            f.write(xfoil_commands)
        
        try:
            # Run XFOIL
            result = subprocess.run([self.xfoil_path], 
                                  input=open(script_file).read(),
                                  text=True, capture_output=True, 
                                  timeout=30, cwd=self.work_dir)
            
            if os.path.exists(output_file):
                return self._parse_xfoil_output(output_file)
            else:
                logger.warning("XFOIL analysis failed, using fallback calculation")
                return self._fallback_analysis(upper_curve, lower_curve)
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"XFOIL execution failed: {e}, using fallback")
            return self._fallback_analysis(upper_curve, lower_curve)
    
    def _write_airfoil_coordinates(self, upper: np.ndarray, lower: np.ndarray, filename: str):
        """Write airfoil coordinates in XFOIL format"""
        # Combine coordinates (upper surface + lower surface reversed)
        coords = np.vstack([upper, lower[::-1]])
        
        with open(filename, 'w') as f:
            f.write("Bio-inspired Airfoil\n")
            for x, y in coords:
                f.write(f"{x:.6f} {y:.6f}\n")
    
    def _parse_xfoil_output(self, output_file: str) -> Dict:
        """Parse XFOIL polar output file"""
        try:
            data = pd.read_csv(output_file, delim_whitespace=True, skiprows=12)
            
            if len(data) == 0:
                raise ValueError("Empty XFOIL output")
            
            # Find optimal operating point (maximum L/D)
            data['L/D'] = data['CL'] / np.maximum(data['CD'], 1e-6)
            optimal_idx = data['L/D'].idxmax()
            
            return {
                'cl_max': float(data['CL'].max()),
                'cd_min': float(data['CD'].min()),
                'cl_cd_max': float(data['L/D'].max()),
                'stall_angle': float(data.loc[data['CL'].idxmax(), 'alpha']),
                'optimal_alpha': float(data.loc[optimal_idx, 'alpha']),
                'optimal_cl': float(data.loc[optimal_idx, 'CL']),
                'optimal_cd': float(data.loc[optimal_idx, 'CD']),
                'polar_data': data.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Failed to parse XFOIL output: {e}")
            return self._fallback_analysis(None, None)
    
    def _fallback_analysis(self, upper_curve: np.ndarray, lower_curve: np.ndarray) -> Dict:
        """Fallback aerodynamic analysis using empirical methods"""
        if upper_curve is None or lower_curve is None:
            return {
                'cl_max': 1.2, 'cd_min': 0.008, 'cl_cd_max': 80.0,
                'stall_angle': 12.0, 'optimal_alpha': 4.0,
                'optimal_cl': 0.8, 'optimal_cd': 0.01
            }
        
        # Calculate geometric properties
        camber_line = (upper_curve[:, 1] + lower_curve[:, 1]) / 2
        thickness_dist = upper_curve[:, 1] - lower_curve[:, 1]
        
        max_camber = np.max(camber_line)
        max_thickness = np.max(thickness_dist)
        
        # Empirical correlations (based on NACA airfoil data)
        cl_max = 0.9 + max_camber * 15 + max_thickness * 2
        cd_min = 0.006 + max_thickness * 0.1 + max_camber**2 * 0.5
        stall_angle = 14 - max_thickness * 20
        
        return {
            'cl_max': min(cl_max, 1.8),
            'cd_min': max(cd_min, 0.005),
            'cl_cd_max': min(cl_max / max(cd_min, 0.005), 150),
            'stall_angle': max(stall_angle, 8.0),
            'optimal_alpha': 4.0 + max_camber * 50,
            'optimal_cl': 0.7 + max_camber * 10,
            'optimal_cd': cd_min
        }


class AdvancedAirfoilEnvironment(gym.Env):
    """Advanced RL environment for airfoil optimization with proper CFD integration"""
    
    def __init__(self, target_birds_df: pd.DataFrame, 
                 use_xfoil: bool = True,
                 reynolds_number: float = 1e6,
                 design_objectives: Dict = None):
        super().__init__()
        
        self.target_birds = target_birds_df.reset_index(drop=True)
        self.current_target_idx = 0
        self.reynolds = reynolds_number
        self.use_xfoil = use_xfoil
        
        # Initialize XFOIL simulator
        if use_xfoil:
            self.simulator = XFOILSimulator()
        else:
            self.simulator = None
            
        # Design objectives (multi-objective optimization)
        self.objectives = design_objectives or {
            'max_lift_drag': 1.0,
            'max_lift': 0.3,
            'min_drag': 0.3,
            'stability': 0.4
        }
        
        # Enhanced action space: control points + shape parameters
        # 14 control point y-coordinates + 6 shape parameters
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        )
        
        # Enhanced observation space: shape + performance + target
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(35,), dtype=np.float32
        )
        
        # Performance tracking
        self.episode_rewards = []
        self.best_performance = -np.inf
        self.convergence_threshold = 0.01
        self.steps_since_improvement = 0
        self.max_steps_without_improvement = 100
        
        self.reset()
    
    def reset(self):
        """Enhanced reset with better target selection"""
        # Cycle through targets or select based on difficulty
        self.current_target_idx = np.random.randint(len(self.target_birds))
        self.current_target = self.target_birds.iloc[self.current_target_idx]
        
        # Reset tracking variables
        self.current_step = 0
        self.steps_since_improvement = 0
        self.episode_performance = []
        
        # Initialize airfoil with smart starting point
        self.current_shape = self._initialize_smart_airfoil()
        
        # Calculate target metrics with proper scaling
        self.target_metrics = self._calculate_enhanced_target_metrics()
        
        return self._get_enhanced_observation()
    
    def step(self, action):
        """Enhanced step function with better reward shaping"""
        self.current_step += 1
        
        # Apply action with constraints
        self._apply_constrained_action(action)
        
        # Evaluate current airfoil performance
        current_metrics = self._evaluate_airfoil_performance()
        
        # Calculate sophisticated reward
        reward = self._calculate_multi_objective_reward(current_metrics)
        
        # Track performance
        self.episode_performance.append(current_metrics)
        
        # Check for improvement
        if reward > self.best_performance:
            self.best_performance = reward
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1
        
        # Enhanced termination conditions
        done = self._check_advanced_termination(current_metrics)
        
        # Enhanced info
        info = {
            'current_metrics': current_metrics,
            'target_metrics': self.target_metrics,
            'performance_improvement': reward - (self.episode_performance[-2]['reward'] if len(self.episode_performance) > 1 else 0),
            'convergence_rate': self._calculate_convergence_rate(),
            'shape_complexity': self._calculate_shape_complexity()
        }
        
        return self._get_enhanced_observation(), reward, done, info
    
    def _initialize_smart_airfoil(self):
        """Initialize airfoil with biological prior knowledge"""
        bird = self.current_target
        
        # Use bird morphology to create intelligent starting point
        wing_aspect = bird['Wing.Length'] / bird['Secondary1']
        efficiency = bird['Hand-Wing.Index'] / 100.0  # Normalize
        pointedness = bird['Kipps.Distance'] / bird['Wing.Length']
        
        # Create biologically-inspired control points
        base_thickness = 0.08 + (bird['Secondary1'] / bird['Wing.Length']) * 0.08
        base_camber = 0.02 + pointedness * 0.04
        
        upper_cp = np.array([
            [0.0, 0.0],
            [0.01, base_thickness * 0.4],
            [0.05, base_thickness * 0.8],
            [0.15, base_camber + base_thickness],
            [0.3, base_camber + base_thickness * 0.9],
            [0.5, base_camber + base_thickness * 0.6],
            [0.7, base_camber + base_thickness * 0.3],
            [0.85, base_camber * 0.5],
            [0.95, base_camber * 0.2],
            [1.0, 0.0]
        ])
        
        lower_cp = np.array([
            [0.0, 0.0],
            [0.01, -base_thickness * 0.2],
            [0.05, -base_thickness * 0.4],
            [0.15, base_camber - base_thickness * 0.8],
            [0.3, base_camber - base_thickness * 0.7],
            [0.5, base_camber - base_thickness * 0.4],
            [0.7, base_camber - base_thickness * 0.2],
            [0.85, -base_thickness * 0.1],
            [0.95, -0.01],
            [1.0, 0.0]
        ])
        
        # Add small random perturbation
        noise = np.random.normal(0, 0.005, upper_cp.shape)
        noise[:, 0] = 0  # Don't perturb x-coordinates
        noise[0, :] = 0  # Fix leading edge
        noise[-1, :] = 0  # Fix trailing edge
        
        return {
            'upper_cp': upper_cp + noise,
            'lower_cp': lower_cp + noise[:len(lower_cp)],
            'parameters': {
                'thickness_scale': 1.0,
                'camber_scale': 1.0,
                'twist': 0.0,
                'leading_edge_radius': 0.01,
                'trailing_edge_thickness': 0.002
            }
        }
    
    def _apply_constrained_action(self, action):
        """Apply action with aerodynamic and geometric constraints"""
        # Split action into control point modifications and parameters
        cp_actions = action[:14] * 0.005  # Smaller learning steps
        param_actions = action[14:] * 0.1
        
        # Update control points with constraints
        upper_cp = self.current_shape['upper_cp'].copy()
        lower_cp = self.current_shape['lower_cp'].copy()
        
        # Modify interior control points only (preserve leading/trailing edges)
        for i in range(1, len(upper_cp) - 1):
            if i < 7:
                upper_cp[i, 1] += cp_actions[i-1]
            else:
                lower_cp[i-7, 1] += cp_actions[i-1]
        
        # Apply geometric constraints
        upper_cp = self._apply_geometric_constraints(upper_cp, 'upper')
        lower_cp = self._apply_geometric_constraints(lower_cp, 'lower')
        
        # Update shape parameters
        params = self.current_shape['parameters']
        params['thickness_scale'] = np.clip(params['thickness_scale'] + param_actions[0], 0.5, 2.0)
        params['camber_scale'] = np.clip(params['camber_scale'] + param_actions[1], 0.1, 3.0)
        params['twist'] = np.clip(params['twist'] + param_actions[2], -5.0, 5.0)
        
        self.current_shape.update({
            'upper_cp': upper_cp,
            'lower_cp': lower_cp,
            'parameters': params
        })
    
    def _apply_geometric_constraints(self, control_points, surface_type):
        """Apply aerodynamic constraints to control points"""
        cp = control_points.copy()
        
        # Ensure monotonic x-coordinates
        for i in range(1, len(cp)):
            if cp[i, 0] <= cp[i-1, 0]:
                cp[i, 0] = cp[i-1, 0] + 0.01
        
        # Thickness constraints
        if surface_type == 'upper':
            cp[1:, 1] = np.maximum(cp[1:, 1], 0.0)  # Upper surface above x-axis
        else:
            cp[1:-1, 1] = np.minimum(cp[1:-1, 1], 0.0)  # Lower surface below x-axis
        
        # Smoothness constraints (prevent sharp discontinuities)
        for i in range(1, len(cp) - 1):
            prev_slope = (cp[i, 1] - cp[i-1, 1]) / (cp[i, 0] - cp[i-1, 0] + 1e-8)
            next_slope = (cp[i+1, 1] - cp[i, 1]) / (cp[i+1, 0] - cp[i, 0] + 1e-8)
            
            # Limit slope changes
            if abs(next_slope - prev_slope) > 2.0:
                cp[i, 1] = cp[i-1, 1] + prev_slope * (cp[i, 0] - cp[i-1, 0])
        
        return cp
    
    def _evaluate_airfoil_performance(self):
        """Comprehensive airfoil performance evaluation"""
        # Generate smooth curves from control points
        upper_curve = self._bezier_curve(self.current_shape['upper_cp'], num_points=100)
        lower_curve = self._bezier_curve(self.current_shape['lower_cp'], num_points=100)
        
        # Apply shape parameters
        upper_curve, lower_curve = self._apply_shape_parameters(upper_curve, lower_curve)
        
        # CFD analysis
        if self.simulator:
            cfd_results = self.simulator.analyze_airfoil(upper_curve, lower_curve, 
                                                       reynolds=self.reynolds)
        else:
            cfd_results = self._advanced_fallback_analysis(upper_curve, lower_curve)
        
        # Geometric analysis
        geometric_metrics = self._calculate_geometric_metrics(upper_curve, lower_curve)
        
        # Combine results
        performance = {
            **cfd_results,
            **geometric_metrics,
            'reynolds': self.reynolds,
            'shape_parameters': self.current_shape['parameters'].copy()
        }
        
        return performance
    
    def _bezier_curve(self, control_points, num_points=100):
        """Generate smooth Bézier curve"""
        n = len(control_points) - 1
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))
        
        for i in range(num_points):
            for j in range(n + 1):
                bernstein = comb(n, j) * (1 - t[i])**(n - j) * t[i]**j
                curve[i] += bernstein * control_points[j]
        
        return curve
    
    def _apply_shape_parameters(self, upper_curve, lower_curve):
        """Apply global shape parameters"""
        params = self.current_shape['parameters']
        
        # Apply thickness scaling
        thickness = (upper_curve[:, 1] - lower_curve[:, 1]) * params['thickness_scale']
        camber_line = (upper_curve[:, 1] + lower_curve[:, 1]) / 2 * params['camber_scale']
        
        upper_modified = upper_curve.copy()
        lower_modified = lower_curve.copy()
        
        upper_modified[:, 1] = camber_line + thickness / 2
        lower_modified[:, 1] = camber_line - thickness / 2
        
        # Apply twist (simple rotation about quarter chord)
        twist_rad = np.radians(params['twist'])
        quarter_chord = 0.25
        
        for i in range(len(upper_modified)):
            x_rel = upper_modified[i, 0] - quarter_chord
            y_rel = upper_modified[i, 1]
            
            upper_modified[i, 0] = quarter_chord + x_rel * np.cos(twist_rad) - y_rel * np.sin(twist_rad)
            upper_modified[i, 1] = x_rel * np.sin(twist_rad) + y_rel * np.cos(twist_rad)
            
            x_rel = lower_modified[i, 0] - quarter_chord
            y_rel = lower_modified[i, 1]
            
            lower_modified[i, 0] = quarter_chord + x_rel * np.cos(twist_rad) - y_rel * np.sin(twist_rad)
            lower_modified[i, 1] = x_rel * np.sin(twist_rad) + y_rel * np.cos(twist_rad)
        
        return upper_modified, lower_modified
    
    def _advanced_fallback_analysis(self, upper_curve, lower_curve):
        """Advanced empirical aerodynamic analysis"""
        # Calculate detailed geometric properties
        x_coords = upper_curve[:, 0]
        camber_line = (upper_curve[:, 1] + lower_curve[:, 1]) / 2
        thickness_dist = upper_curve[:, 1] - lower_curve[:, 1]
        
        # Key geometric parameters
        max_camber = np.max(camber_line)
        max_camber_pos = x_coords[np.argmax(camber_line)]
        max_thickness = np.max(thickness_dist)
        max_thickness_pos = x_coords[np.argmax(thickness_dist)]
        
        # Leading edge radius (approximate)
        le_curvature = self._calculate_leading_edge_curvature(upper_curve, lower_curve)
        
        # Advanced empirical correlations based on Raymer, Anderson, and Abbott & von Doenhoff
        # Lift coefficient slope (per radian)
        a0 = 2 * np.pi * (1 + 0.77 * max_thickness)  # Compressibility correction
        
        # Zero-lift angle
        alpha_L0 = -4 * max_camber * (1 - max_camber_pos) * 180 / np.pi  # degrees
        
        # Maximum lift coefficient
        cl_max = 0.9 + 4.0 * max_camber + 1.2 * max_thickness - 2.0 * max_camber**2
        
        # Profile drag coefficient
        cd_profile = 0.006 + 0.02 * max_thickness**2 + 0.1 * max_camber**2
        
        # Pressure gradient parameter (affects boundary layer)
        pressure_gradient = self._estimate_pressure_gradient(upper_curve, lower_curve)
        cd_profile *= (1 + 0.2 * pressure_gradient)
        
        # Moment coefficient about quarter chord
        cm_quarter = -0.05 - 0.1 * max_camber * (max_camber_pos - 0.25)
        
        # Stall characteristics
        stall_angle = 16 - 20 * max_thickness + 10 * max_camber
        stall_angle = np.clip(stall_angle, 8, 18)
        
        return {
            'cl_max': float(np.clip(cl_max, 0.8, 2.0)),
            'cd_min': float(np.clip(cd_profile, 0.005, 0.05)),
            'cl_cd_max': float(np.clip(cl_max / cd_profile, 50, 200)),
            'stall_angle': float(stall_angle),
            'optimal_alpha': float(4.0 + max_camber * 30),
            'optimal_cl': float(0.5 + 8 * max_camber),
            'optimal_cd': float(cd_profile),
            'cm_quarter': float(cm_quarter),
            'alpha_L0': float(alpha_L0)
        }
    
    def _calculate_geometric_metrics(self, upper_curve, lower_curve):
        """Calculate detailed geometric metrics"""
        # Camber line and thickness distribution
        camber_line = (upper_curve[:, 1] + lower_curve[:, 1]) / 2
        thickness_dist = upper_curve[:, 1] - lower_curve[:, 1]
        
        return {
            'max_camber': float(np.max(camber_line)),
            'max_thickness': float(np.max(thickness_dist)),
            'camber_position': float(upper_curve[np.argmax(camber_line), 0]),
            'thickness_position': float(upper_curve[np.argmax(thickness_dist), 0]),
            'leading_edge_radius': float(self._calculate_leading_edge_curvature(upper_curve, lower_curve)),
            'trailing_edge_angle': float(self._calculate_trailing_edge_angle(upper_curve, lower_curve)),
            'area': float(self._calculate_area(upper_curve, lower_curve)),
            'chord_length': float(upper_curve[-1, 0] - upper_curve[0, 0])
        }
    
    def _calculate_leading_edge_curvature(self, upper_curve, lower_curve):
        """Estimate leading edge radius of curvature"""
        # Use first few points to estimate curvature
        if len(upper_curve) < 5:
            return 0.01
        
        # Calculate curvature using finite differences
        x1, y1 = upper_curve[1] - upper_curve[0]
        x2, y2 = upper_curve[2] - upper_curve[1]
        
        # Approximate radius of curvature
        denominator = abs(x1*y2 - x2*y1)
        if denominator < 1e-6:
            return 0.01
        
        radius = ((x1**2 + y1**2)**(1.5)) / denominator
        return min(radius, 0.1)  # Reasonable bounds
    
    def _calculate_trailing_edge_angle(self, upper_curve, lower_curve):
        """Calculate trailing edge angle"""
        if len(upper_curve) < 3:
            return 0.0
        
        # Use last few points
        upper_slope = (upper_curve[-1, 1] - upper_curve[-3, 1]) / (upper_curve[-1, 0] - upper_curve[-3, 0] + 1e-8)
        lower_slope = (lower_curve[-1, 1] - lower_curve[-3, 1]) / (lower_curve[-1, 0] - lower_curve[-3, 0] + 1e-8)
        
        angle = abs(np.arctan(upper_slope) - np.arctan(lower_slope))
        return float(np.degrees(angle))
    
    def _calculate_area(self, upper_curve, lower_curve):
        """Calculate airfoil cross-sectional area"""
        # Trapezoidal integration
        x_coords = upper_curve[:, 0]
        thickness_dist = upper_curve[:, 1] - lower_curve[:, 1]
        
        area = np.trapz(thickness_dist, x_coords)
        return abs(area)
    
    def _estimate_pressure_gradient(self, upper_curve, lower_curve):
        """Estimate adverse pressure gradient parameter"""
        # Simple approximation based on surface curvature
        upper_curvature = np.gradient(np.gradient(upper_curve[:, 1]))
        pressure_gradient = np.mean(np.abs(upper_curvature))
        return float(np.clip(pressure_gradient, 0, 2))
    
    def _calculate_enhanced_target_metrics(self):
        """Calculate sophisticated target metrics from bird morphology"""
        bird = self.current_target
        
        # Morphological ratios
        aspect_ratio = (bird['Wing.Length'] ** 2) / (bird['Wing.Length'] * bird['Secondary1'])
        hand_wing_ratio = bird['Hand-Wing.Index'] / 100.0
        kipps_ratio = bird['Kipps.Distance'] / bird['Wing.Length']
        secondary_ratio = bird['Secondary1'] / bird['Wing.Length']
        
        # Flight regime classification (based on morphology)
        if hand_wing_ratio > 0.4 and kipps_ratio > 0.4:
            flight_regime = 'high_speed'
            target_cl_cd = 80 + hand_wing_ratio * 40
            target_cl_max = 1.2 + kipps_ratio * 0.4
        elif aspect_ratio > 8 and secondary_ratio > 0.6:
            flight_regime = 'soaring'
            target_cl_cd = 120 + aspect_ratio * 5
            target_cl_max = 1.4 + secondary_ratio * 0.3
        elif secondary_ratio > 0.7:
            flight_regime = 'maneuvering'
            target_cl_cd = 60 + secondary_ratio * 30
            target_cl_max = 1.6 + secondary_ratio * 0.5
        else:
            flight_regime = 'general'
            target_cl_cd = 70
            target_cl_max = 1.3
        
        return {
            'target_cl_cd': float(target_cl_cd),
            'target_cl_max': float(target_cl_max),
            'target_cd_min': float(0.008 + hand_wing_ratio * 0.002),
            'target_stall_angle': float(14 + kipps_ratio * 2),
            'flight_regime': flight_regime,
            'aspect_ratio': float(aspect_ratio),
            'morphological_score': float((hand_wing_ratio + kipps_ratio + secondary_ratio) / 3)
        }
    
    def _calculate_multi_objective_reward(self, current_metrics):
        """Advanced multi-objective reward function"""
        target = self.target_metrics
        
        # Individual objective scores
        objectives = {}
        
        # 1. Lift-to-drag ratio objective
        cl_cd_current = current_metrics.get('cl_cd_max', 50)
        cl_cd_target = target['target_cl_cd']
        cl_cd_error = abs(cl_cd_current - cl_cd_target) / cl_cd_target
        objectives['lift_drag'] = np.exp(-cl_cd_error) * self.objectives['max_lift_drag']
        
        # 2. Maximum lift coefficient objective
        cl_max_current = current_metrics.get('cl_max', 1.0)
        cl_max_target = target['target_cl_max']
        cl_max_error = abs(cl_max_current - cl_max_target) / cl_max_target
        objectives['max_lift'] = np.exp(-cl_max_error) * self.objectives['max_lift']
        
        # 3. Minimum drag objective
        cd_min_current = current_metrics.get('cd_min', 0.02)
        cd_min_target = target['target_cd_min']
        cd_min_error = abs(cd_min_current - cd_min_target) / cd_min_target
        objectives['min_drag'] = np.exp(-cd_min_error) * self.objectives['min_drag']
        
        # 4. Stability objective (moment coefficient and stall behavior)
        stall_current = current_metrics.get('stall_angle', 12)
        stall_target = target['target_stall_angle']
        stall_error = abs(stall_current - stall_target) / stall_target
        
        cm_penalty = abs(current_metrics.get('cm_quarter', 0)) * 2  # Penalize large moments
        objectives['stability'] = (np.exp(-stall_error) - cm_penalty) * self.objectives['stability']
        
        # 5. Bonus objectives
        # Geometric efficiency bonus
        thickness_ratio = current_metrics.get('max_thickness', 0.1)
        if 0.08 <= thickness_ratio <= 0.15:  # Optimal thickness range
            objectives['geometric_bonus'] = 0.2
        else:
            objectives['geometric_bonus'] = -0.1
        
        # Smoothness bonus (penalize sharp changes)
        shape_complexity = self._calculate_shape_complexity()
        objectives['smoothness_bonus'] = max(0, 0.1 - shape_complexity * 0.5)
        
        # Biological plausibility bonus
        bio_score = self._calculate_biological_plausibility(current_metrics)
        objectives['bio_bonus'] = bio_score * 0.15
        
        # Combine objectives
        total_reward = sum(objectives.values())
        
        # Progressive reward scaling (encourage improvement)
        if hasattr(self, 'previous_reward'):
            improvement = total_reward - self.previous_reward
            total_reward += improvement * 0.1  # Improvement bonus
        
        self.previous_reward = total_reward
        
        # Add exploration bonus early in training
        if self.current_step < 50:
            exploration_bonus = 0.05 * np.random.random()
            total_reward += exploration_bonus
        
        return float(total_reward)
    
    def _calculate_shape_complexity(self):
        """Calculate shape complexity metric"""
        upper_cp = self.current_shape['upper_cp']
        lower_cp = self.current_shape['lower_cp']
        
        # Calculate curvature changes
        upper_curvature = 0
        lower_curvature = 0
        
        for i in range(1, len(upper_cp) - 1):
            # Second derivative approximation
            d2y_dx2 = (upper_cp[i+1, 1] - 2*upper_cp[i, 1] + upper_cp[i-1, 1])
            upper_curvature += abs(d2y_dx2)
            
        for i in range(1, len(lower_cp) - 1):
            d2y_dx2 = (lower_cp[i+1, 1] - 2*lower_cp[i, 1] + lower_cp[i-1, 1])
            lower_curvature += abs(d2y_dx2)
        
        return (upper_curvature + lower_curvature) / (len(upper_cp) + len(lower_cp))
    
    def _calculate_biological_plausibility(self, metrics):
        """Reward biologically plausible designs"""
        # Check if metrics fall within observed biological ranges
        bio_score = 0
        
        # Thickness range (most birds: 8-18%)
        thickness = metrics.get('max_thickness', 0.1)
        if 0.08 <= thickness <= 0.18:
            bio_score += 0.3
        
        # Camber range (0-8%)
        camber = metrics.get('max_camber', 0.02)
        if 0.0 <= camber <= 0.08:
            bio_score += 0.3
        
        # L/D ratio range (typical bird range: 10-150)
        ld_ratio = metrics.get('cl_cd_max', 70)
        if 10 <= ld_ratio <= 150:
            bio_score += 0.4
        
        return bio_score
    
    def _check_advanced_termination(self, current_metrics):
        """Advanced termination conditions"""
        # 1. Convergence check
        if len(self.episode_performance) > 10:
            recent_rewards = [p.get('reward', 0) for p in self.episode_performance[-10:]]
            if np.std(recent_rewards) < self.convergence_threshold:
                return True
        
        # 2. Performance target reached
        target_achieved = (
            abs(current_metrics.get('cl_cd_max', 0) - self.target_metrics['target_cl_cd']) / 
            self.target_metrics['target_cl_cd'] < 0.05
        )
        if target_achieved:
            return True
        
        # 3. Maximum steps without improvement
        if self.steps_since_improvement > self.max_steps_without_improvement:
            return True
        
        # 4. Maximum episode length
        if self.current_step > 300:
            return True
        
        return False
    
    def _calculate_convergence_rate(self):
        """Calculate convergence rate metric"""
        if len(self.episode_performance) < 5:
            return 0.0
        
        recent_rewards = [p.get('reward', 0) for p in self.episode_performance[-5:]]
        return float(np.std(recent_rewards))
    
    def _get_enhanced_observation(self):
        """Get comprehensive observation state"""
        # Current shape representation
        upper_cp_flat = self.current_shape['upper_cp'][:, 1]  # y-coordinates only
        lower_cp_flat = self.current_shape['lower_cp'][:, 1]
        
        # Shape parameters
        params = self.current_shape['parameters']
        param_values = np.array([
            params['thickness_scale'],
            params['camber_scale'], 
            params['twist'],
            params['leading_edge_radius'],
            params['trailing_edge_thickness']
        ])
        
        # Target information
        target_info = np.array([
            self.target_metrics['target_cl_cd'] / 100,  # Normalize
            self.target_metrics['target_cl_max'],
            self.target_metrics['target_cd_min'] * 100,  # Scale up
            self.target_metrics['target_stall_angle'] / 20,
            self.target_metrics['morphological_score']
        ])
        
        # Current performance (if available)
        if hasattr(self, 'last_performance'):
            perf_info = np.array([
                self.last_performance.get('cl_cd_max', 70) / 100,
                self.last_performance.get('cl_max', 1.2),
                self.last_performance.get('cd_min', 0.01) * 100,
                self.last_performance.get('stall_angle', 14) / 20
            ])
        else:
            perf_info = np.zeros(4)
        
        # Progress information
        progress_info = np.array([
            self.current_step / 300,  # Normalized step
            self.steps_since_improvement / 100,  # Normalized
            len(self.episode_performance) / 300
        ])
        
        # Combine all observations (ensure size matches observation_space)
        obs = np.concatenate([
            upper_cp_flat[:8],  # First 8 upper control points
            lower_cp_flat[:8],  # First 8 lower control points
            param_values,       # 5 parameters
            target_info,        # 5 target values
            perf_info,          # 4 performance values
            progress_info       # 3 progress values
        ])
        
        return obs.astype(np.float32)


class PerformanceTrackingCallback(BaseCallback):
    """Custom callback for tracking training performance"""
    
    def __init__(self, eval_env, eval_freq=1000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations_results = []
        self.evaluations_timesteps = []
        
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            # Evaluate current model
            episode_rewards = []
            episode_lengths = []
            
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                episode_reward = 0
                episode_length = 0
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            self.evaluations_results.append(episode_rewards)
            self.evaluations_timesteps.append(self.num_timesteps)
            
            if self.verbose > 0:
                print(f"Eval at {self.num_timesteps}: "
                      f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}, "
                      f"Mean length: {mean_length:.1f}")
        
        return True


class AdvancedBioInspiredOptimizer:
    """Main orchestrator for bio-inspired airfoil optimization"""
    
    def __init__(self, bird_data_path: str, results_dir: str = "./results"):
        self.bird_data_path = bird_data_path
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Load and process data
        self.load_bird_data()
        self.analyze_bird_morphology()
        self.select_optimization_targets()
        
    def load_bird_data(self):
        """Load and preprocess AVONET bird data"""
        try:
            self.raw_bird_data = pd.read_csv(self.bird_data_path, encoding='ISO-8859-1')
            logger.info(f"Loaded {len(self.raw_bird_data)} bird species")
            
            # Filter for complete morphological data
            required_cols = ['Species1', 'Wing.Length', 'Secondary1', 'Kipps.Distance', 
                           'Hand-Wing.Index', 'Tail.Length']
            
            self.bird_data = self.raw_bird_data[required_cols].dropna()
            self.bird_data = self.bird_data.rename(columns={'Species1': 'species'})
            
            logger.info(f"Using {len(self.bird_data)} species with complete data")
            
        except Exception as e:
            logger.warning(f"Could not load bird data: {e}")
            self.bird_data = self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate comprehensive sample bird data"""
        np.random.seed(42)
        
        # Define bird archetypes with realistic morphological ranges
        archetypes = {
            'Hummingbird': {'wl': (40, 60), 's1': (20, 35), 'kd': (15, 25), 'hwi': (35, 50), 'tl': (15, 25)},
            'Swift': {'wl': (150, 200), 's1': (70, 100), 'kd': (70, 120), 'hwi': (45, 65), 'tl': (60, 90)},
            'Swallow': {'wl': (100, 140), 's1': (60, 85), 'kd': (40, 65), 'hwi': (40, 55), 'tl': (50, 80)},
            'Hawk': {'wl': (300, 450), 's1': (180, 260), 'kd': (120, 200), 'hwi': (25, 40), 'tl': (150, 220)},
            'Eagle': {'wl': (500, 700), 's1': (300, 450), 'kd': (200, 350), 'hwi': (20, 35), 'tl': (200, 300)},
            'Albatross': {'wl': (600, 800), 's1': (250, 350), 'kd': (350, 500), 'hwi': (50, 70), 'tl': (180, 250)},
            'Falcon': {'wl': (250, 350), 's1': (140, 200), 'kd': (110, 180), 'hwi': (55, 75), 'tl': (120, 180)},
            'Crow': {'wl': (250, 320), 's1': (170, 220), 'kd': (80, 130), 'hwi': (25, 40), 'tl': (150, 200)}
        }
        
        bird_list = []
        for archetype, ranges in archetypes.items():
            for i in range(50):  # 50 variants per archetype
                bird = {
                    'species': f"{archetype}_{i+1:03d}",
                    'Wing.Length': np.random.uniform(*ranges['wl']),
                    'Secondary1': np.random.uniform(*ranges['s1']),
                    'Kipps.Distance': np.random.uniform(*ranges['kd']),
                    'Hand-Wing.Index': np.random.uniform(*ranges['hwi']),
                    'Tail.Length': np.random.uniform(*ranges['tl'])
                }
                bird_list.append(bird)
        
        return pd.DataFrame(bird_list)
    
    def analyze_bird_morphology(self):
        """Comprehensive morphological analysis"""
        logger.info("Analyzing bird wing morphology...")
        
        # Calculate morphological indices
        self.bird_data['Aspect_Ratio'] = (
            self.bird_data['Wing.Length']**2 / 
            (self.bird_data['Wing.Length'] * self.bird_data['Secondary1'])
        )
        self.bird_data['Wing_Loading_Index'] = (
            self.bird_data['Wing.Length'] / self.bird_data['Secondary1']
        )
        self.bird_data['Pointedness_Index'] = (
            self.bird_data['Kipps.Distance'] / self.bird_data['Wing.Length']
        )
        self.bird_data['Secondary_Ratio'] = (
            self.bird_data['Secondary1'] / self.bird_data['Wing.Length']
        )
        
        # Performance estimates
        self.bird_data['Estimated_Speed_Performance'] = (
            self.bird_data['Hand-Wing.Index'] * self.bird_data['Pointedness_Index']
        )
        self.bird_data['Estimated_Soaring_Performance'] = (
            self.bird_data['Aspect_Ratio'] * self.bird_data['Hand-Wing.Index'] / 100
        )
        self.bird_data['Estimated_Maneuver_Performance'] = (
            self.bird_data['Secondary_Ratio'] / self.bird_data['Wing_Loading_Index']
        )
        
        # Create comprehensive visualization
        self.plot_morphological_analysis()
        
    def plot_morphological_analysis(self):
        """Create comprehensive morphological analysis plots"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Comprehensive Bird Wing Morphology Analysis', fontsize=16)
        
        # Plot 1: Wing length vs aspect ratio
        ax = axes[0, 0]
        scatter = ax.scatter(self.bird_data['Wing.Length'], self.bird_data['Aspect_Ratio'],
                           c=self.bird_data['Hand-Wing.Index'], cmap='viridis', alpha=0.7)
        ax.set_xlabel('Wing Length (mm)')
        ax.set_ylabel('Aspect Ratio')
        ax.set_title('Wing Geometry Relationships')
        plt.colorbar(scatter, ax=ax, label='Hand-Wing Index')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Performance triangle
        ax = axes[0, 1]
        scatter = ax.scatter(self.bird_data['Estimated_Speed_Performance'],
                           self.bird_data['Estimated_Soaring_Performance'],
                           c=self.bird_data['Estimated_Maneuver_Performance'], 
                           cmap='plasma', alpha=0.7, s=50)
        ax.set_xlabel('Speed Performance Index')
        ax.set_ylabel('Soaring Performance Index')
        ax.set_title('Flight Performance Space')
        plt.colorbar(scatter, ax=ax, label='Maneuver Performance')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Hand-wing index distribution
        ax = axes[0, 2]
        ax.hist(self.bird_data['Hand-Wing.Index'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(self.bird_data['Hand-Wing.Index'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {self.bird_data["Hand-Wing.Index"].mean():.1f}')
        ax.set_xlabel('Hand-Wing Index')
        ax.set_ylabel('Frequency')
        ax.set_title('Hand-Wing Index Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Kipps distance vs wing pointedness
        ax = axes[1, 0]
        scatter = ax.scatter(self.bird_data['Kipps.Distance'], self.bird_data['Pointedness_Index'],
                           c=self.bird_data['Wing.Length'], cmap='coolwarm', alpha=0.7)
        ax.set_xlabel('Kipps Distance (mm)')
        ax.set_ylabel('Pointedness Index')
        ax.set_title('Wing Tip Characteristics')
        plt.colorbar(scatter, ax=ax, label='Wing Length (mm)')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Secondary feather analysis
        ax = axes[1, 1]
        scatter = ax.scatter(self.bird_data['Secondary1'], self.bird_data['Secondary_Ratio'],
                           c=self.bird_data['Aspect_Ratio'], cmap='Spectral', alpha=0.7)
        ax.set_xlabel('Secondary1 Length (mm)')
        ax.set_ylabel('Secondary Ratio')
        ax.set_title('Secondary Feather Development')
        plt.colorbar(scatter, ax=ax, label='Aspect Ratio')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Performance correlation matrix
        ax = axes[1, 2]
        perf_cols = ['Estimated_Speed_Performance', 'Estimated_Soaring_Performance', 
                    'Estimated_Maneuver_Performance', 'Hand-Wing.Index']
        corr_matrix = self.bird_data[perf_cols].corr()
        im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_xticks(range(len(perf_cols)))
        ax.set_yticks(range(len(perf_cols)))
        ax.set_xticklabels([col.replace('Estimated_', '').replace('_', ' ') for col in perf_cols], rotation=45)
        ax.set_yticklabels([col.replace('Estimated_', '').replace('_', ' ') for col in perf_cols])
        ax.set_title('Performance Correlations')
        plt.colorbar(im, ax=ax)
        
        # Add correlation values
        for i in range(len(perf_cols)):
            for j in range(len(perf_cols)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center',
                       color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        # Plot 7-9: Top performers in each category
        for idx, (performance_type, ax) in enumerate(zip(
            ['Estimated_Speed_Performance', 'Estimated_Soaring_Performance', 'Estimated_Maneuver_Performance'],
            [axes[2, 0], axes[2, 1], axes[2, 2]]
        )):
            top_performers = self.bird_data.nlargest(10, performance_type)
            
            bars = ax.barh(range(len(top_performers)), top_performers[performance_type])
            ax.set_yticks(range(len(top_performers)))
            ax.set_yticklabels([species[:15] + '...' if len(species) > 15 else species 
                              for species in top_performers['species']], fontsize=8)
            ax.set_xlabel('Performance Index')
            ax.set_title(f'Top {performance_type.replace("Estimated_", "").replace("_", " ")} Performers')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Color bars by performance
            for i, bar in enumerate(bars):
                bar.set_color(plt.cm.viridis(i / len(bars)))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'optimization_targets.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_optimization_models(self, algorithm='PPO', total_timesteps=100000, 
                                n_parallel_envs=4, use_xfoil=True):
        """Train RL models for airfoil optimization"""
        logger.info(f"Training {algorithm} model with {total_timesteps} timesteps...")
        
        # Create training environment
        def make_env():
            env = AdvancedAirfoilEnvironment(
                target_birds_df=self.target_birds,
                use_xfoil=use_xfoil,
                reynolds_number=1e6
            )
            return Monitor(env)
        
        # Create vectorized environment for parallel training
        if n_parallel_envs > 1:
            env = SubprocVecEnv([make_env for _ in range(n_parallel_envs)])
        else:
            env = DummyVecEnv([make_env])
        
        # Create evaluation environment
        eval_env = make_env()
        
        # Configure training algorithm
        if algorithm == 'PPO':
            model = PPO(
                'MlpPolicy', 
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                normalize_advantage=True,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log=os.path.join(self.results_dir, 'tensorboard'),
                verbose=1
            )
        elif algorithm == 'SAC':
            model = SAC(
                'MlpPolicy',
                env,
                learning_rate=3e-4,
                buffer_size=100000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef='auto',
                tensorboard_log=os.path.join(self.results_dir, 'tensorboard'),
                verbose=1
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Setup performance tracking
        callback = PerformanceTrackingCallback(
            eval_env=eval_env,
            eval_freq=2000,
            n_eval_episodes=5,
            verbose=1
        )
        
        # Train the model
        start_time = time.time()
        model.learn(total_timesteps=total_timesteps, callback=callback)
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save the trained model
        model_path = os.path.join(self.results_dir, f'trained_model_{algorithm.lower()}')
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Plot training progress
        self.plot_training_progress(callback)
        
        return model, callback
    
    def plot_training_progress(self, callback):
        """Plot comprehensive training progress"""
        if not callback.evaluations_timesteps:
            logger.warning("No evaluation data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Training Progress', fontsize=16)
        
        timesteps = callback.evaluations_timesteps
        
        # Plot 1: Mean reward over time
        ax = axes[0, 0]
        mean_rewards = [np.mean(rewards) for rewards in callback.evaluations_results]
        std_rewards = [np.std(rewards) for rewards in callback.evaluations_results]
        
        ax.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
        ax.fill_between(timesteps, 
                       np.array(mean_rewards) - np.array(std_rewards),
                       np.array(mean_rewards) + np.array(std_rewards),
                       alpha=0.3, color='blue')
        
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Reward distribution evolution
        ax = axes[0, 1]
        if len(callback.evaluations_results) > 5:
            # Show reward distribution for first, middle, and last evaluations
            early_rewards = callback.evaluations_results[0]
            mid_rewards = callback.evaluations_results[len(callback.evaluations_results)//2]
            late_rewards = callback.evaluations_results[-1]
            
            ax.hist(early_rewards, alpha=0.5, bins=10, label='Early Training', color='red')
            ax.hist(mid_rewards, alpha=0.5, bins=10, label='Mid Training', color='yellow')
            ax.hist(late_rewards, alpha=0.5, bins=10, label='Late Training', color='green')
            
            ax.set_xlabel('Episode Reward')
            ax.set_ylabel('Frequency')
            ax.set_title('Reward Distribution Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Performance improvement rate
        ax = axes[1, 0]
        if len(mean_rewards) > 1:
            improvement_rate = np.diff(mean_rewards)
            ax.plot(timesteps[1:], improvement_rate, 'g-', linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Reward Improvement')
            ax.set_title('Learning Rate (Reward Improvement)')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Best performance tracking
        ax = axes[1, 1]
        best_rewards = []
        current_best = -np.inf
        
        for rewards in callback.evaluations_results:
            episode_best = np.max(rewards)
            if episode_best > current_best:
                current_best = episode_best
            best_rewards.append(current_best)
        
        ax.plot(timesteps, best_rewards, 'r-', linewidth=2, marker='o', 
                markersize=4, label='Best Reward')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Best Episode Reward')
        ax.set_title('Best Performance Tracking')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_progress.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_trained_model(self, model, n_episodes=20, detailed_analysis=True):
        """Comprehensive evaluation of trained model"""
        logger.info(f"Evaluating trained model over {n_episodes} episodes...")
        
        # Create evaluation environment
        eval_env = AdvancedAirfoilEnvironment(
            target_birds_df=self.target_birds,
            use_xfoil=True,
            reynolds_number=1e6
        )
        
        evaluation_results = []
        airfoil_designs = []
        
        for episode in range(n_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Track episode progression
            episode_data = {
                'episode': episode,
                'target_bird': eval_env.current_target['species'],
                'target_strategy': eval_env.current_target.get('optimization_strategy', 'unknown'),
                'rewards': [],
                'actions': [],
                'observations': [],
                'performance_metrics': []
            }
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Store detailed episode data
                episode_data['rewards'].append(float(reward))
                episode_data['actions'].append(action.tolist())
                episode_data['observations'].append(obs.tolist())
                episode_data['performance_metrics'].append(info['current_metrics'])
            
            # Final episode statistics
            episode_data.update({
                'total_reward': episode_reward,
                'episode_length': episode_length,
                'final_performance': info['current_metrics'],
                'target_achieved': info['current_metrics']['cl_cd_max'] > 
                                 info['target_metrics']['target_cl_cd'] * 0.9
            })
            
            evaluation_results.append(episode_data)
            
            # Store final airfoil design
            final_airfoil = {
                'episode': episode,
                'target_species': eval_env.current_target['species'],
                'upper_control_points': eval_env.current_shape['upper_cp'].tolist(),
                'lower_control_points': eval_env.current_shape['lower_cp'].tolist(),
                'shape_parameters': eval_env.current_shape['parameters'],
                'performance_metrics': info['current_metrics'],
                'target_metrics': info['target_metrics']
            }
            airfoil_designs.append(final_airfoil)
        
        # Save detailed results
        results_df = pd.DataFrame([{
            'episode': r['episode'],
            'target_bird': r['target_bird'],
            'target_strategy': r['target_strategy'], 
            'total_reward': r['total_reward'],
            'episode_length': r['episode_length'],
            'target_achieved': r['target_achieved'],
            'final_cl_cd': r['final_performance']['cl_cd_max'],
            'target_cl_cd': r['final_performance'].get('target_cl_cd', 0),
            'final_cl_max': r['final_performance']['cl_max'],
            'final_cd_min': r['final_performance']['cd_min']
        } for r in evaluation_results])
        
        results_df.to_csv(os.path.join(self.results_dir, 'evaluation_results.csv'), index=False)
        
        # Create comprehensive evaluation plots
        if detailed_analysis:
            self.plot_evaluation_analysis(evaluation_results, results_df)
        
        # Print summary statistics
        self.print_evaluation_summary(results_df)
        
        return evaluation_results, airfoil_designs, results_df
    
    def plot_evaluation_analysis(self, evaluation_results, results_df):
        """Create comprehensive evaluation analysis plots"""
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
        
        # Plot 1: Performance vs Target by Strategy
        ax1 = fig.add_subplot(gs[0, :2])
        strategies = results_df['target_strategy'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
        
        for i, strategy in enumerate(strategies):
            strategy_data = results_df[results_df['target_strategy'] == strategy]
            ax1.scatter(strategy_data['target_cl_cd'], strategy_data['final_cl_cd'],
                       c=[colors[i]], s=80, alpha=0.7, label=strategy.replace('_', ' ').title(),
                       edgecolor='black', linewidth=0.5)
        
        # Perfect performance line
        min_val = min(results_df['target_cl_cd'].min(), results_df['final_cl_cd'].min())
        max_val = max(results_df['target_cl_cd'].max(), results_df['final_cl_cd'].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Match')
        
        ax1.set_xlabel('Target L/D Ratio')
        ax1.set_ylabel('Achieved L/D Ratio')
        ax1.set_title('Performance Achievement by Strategy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Reward distribution by strategy
        ax2 = fig.add_subplot(gs[0, 2:])
        strategy_rewards = [results_df[results_df['target_strategy'] == s]['total_reward'].values 
                           for s in strategies]
        
        bp = ax2.boxplot(strategy_rewards, labels=[s.replace('_', ' ').title() for s in strategies],
                        patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Optimization Strategy')
        ax2.set_ylabel('Total Episode Reward')
        ax2.set_title('Reward Distribution by Strategy')
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3-6: Best airfoil designs from each strategy
        for i, strategy in enumerate(strategies[:4]):
            ax = fig.add_subplot(gs[1 + i//2, (i%2)*2:(i%2)*2+2])
            
            # Find best performing episode for this strategy
            strategy_data = results_df[results_df['target_strategy'] == strategy]
            if len(strategy_data) == 0:
                continue
                
            best_episode_idx = strategy_data['total_reward'].idxmax()
            best_episode = evaluation_results[best_episode_idx]
            
            # Find corresponding airfoil design
            best_airfoil = None
            for design in airfoil_designs if 'airfoil_designs' in locals() else []:
                if design['episode'] == best_episode['episode']:
                    best_airfoil = design
                    break
            
            if best_airfoil:
                # Plot airfoil shape
                upper_cp = np.array(best_airfoil['upper_control_points'])
                lower_cp = np.array(best_airfoil['lower_control_points'])
                
                # Generate curves
                env = AdvancedAirfoilEnvironment(self.target_birds)  # Temporary env for curve generation
                upper_curve = env._bezier_curve(upper_cp, 100)
                lower_curve = env._bezier_curve(lower_cp, 100)
                
                # Plot airfoil
                ax.plot(upper_curve[:, 0], upper_curve[:, 1], 'b-', linewidth=2, label='Upper')
                ax.plot(lower_curve[:, 0], lower_curve[:, 1], 'r-', linewidth=2, label='Lower')
                ax.fill_between(upper_curve[:, 0], upper_curve[:, 1], lower_curve[:, 1], 
                               alpha=0.3, color='lightblue')
                
                # Plot control points
                ax.plot(upper_cp[:, 0], upper_cp[:, 1], 'bo--', alpha=0.5, markersize=4)
                ax.plot(lower_cp[:, 0], lower_cp[:, 1], 'ro--', alpha=0.5, markersize=4)
                
                ax.set_xlim(-0.05, 1.05)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                
                # Title with performance metrics
                perf = best_airfoil['performance_metrics']
                title = f"{strategy.replace('_', ' ').title()} Strategy\n"
                title += f"L/D: {perf.get('cl_cd_max', 0):.1f}, "
                title += f"CL_max: {perf.get('cl_max', 0):.2f}"
                ax.set_title(title, fontsize=10)
                
                if i == 0:
                    ax.legend(fontsize=8)
        
        plt.suptitle('Comprehensive Evaluation Analysis', fontsize=16)
        plt.savefig(os.path.join(self.results_dir, 'evaluation_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_evaluation_summary(self, results_df):
        """Print comprehensive evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Total Episodes: {len(results_df)}")
        print(f"Mean Episode Reward: {results_df['total_reward'].mean():.2f} ± {results_df['total_reward'].std():.2f}")
        print(f"Success Rate (90%+ target achievement): {results_df['target_achieved'].mean()*100:.1f}%")
        print(f"Mean Episode Length: {results_df['episode_length'].mean():.1f} steps")
        
        print(f"\nPerformance Metrics:")
        print(f"  Mean Achieved L/D: {results_df['final_cl_cd'].mean():.1f}")
        print(f"  Mean Target L/D: {results_df['target_cl_cd'].mean():.1f}")
        print(f"  L/D Achievement Rate: {(results_df['final_cl_cd']/results_df['target_cl_cd']).mean()*100:.1f}%")
        
        print(f"\nBy Strategy:")
        for strategy in results_df['target_strategy'].unique():
            strategy_data = results_df[results_df['target_strategy'] == strategy]
            print(f"  {strategy.replace('_', ' ').title()}:")
            print(f"    Episodes: {len(strategy_data)}")
            print(f"    Mean Reward: {strategy_data['total_reward'].mean():.2f}")
            print(f"    Success Rate: {strategy_data['target_achieved'].mean()*100:.1f}%")
            print(f"    Mean L/D: {strategy_data['final_cl_cd'].mean():.1f}")
        
        print("="*60)
    
    def run_complete_optimization(self, algorithm='PPO', timesteps=100000, 
                                use_xfoil=True, n_parallel_envs=4):
        """Run the complete bio-inspired optimization pipeline"""
        logger.info("Starting complete bio-inspired airfoil optimization pipeline...")
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                 BIO-INSPIRED AIRFOIL OPTIMIZATION           ║
║                                                              ║
║  🦅 Bird Morphology Analysis                                ║
║  🧠 Reinforcement Learning Training                         ║
║  ✈️  Airfoil Performance Optimization                       ║
║  📊 Comprehensive Evaluation                                ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Step 1: Data analysis already completed in initialization
        print("✅ Bird morphology analysis completed")
        
        # Step 2: Train RL model
        print(f"🧠 Training {algorithm} model...")
        trained_model, training_callback = self.train_optimization_models(
            algorithm=algorithm,
            total_timesteps=timesteps,
            n_parallel_envs=n_parallel_envs,
            use_xfoil=use_xfoil
        )
        print("✅ RL training completed")
        
        # Step 3: Comprehensive evaluation
        print("📊 Evaluating trained model...")
        evaluation_results, airfoil_designs, results_df = self.evaluate_trained_model(
            trained_model, 
            n_episodes=30,
            detailed_analysis=True
        )
        print("✅ Evaluation completed")
        
        # Step 4: Generate final report
        self.generate_optimization_report(trained_model, evaluation_results, results_df)
        print("✅ Optimization report generated")
        
        print(f"\n🎉 Complete optimization pipeline finished!")
        print(f"📁 Results saved to: {self.results_dir}")
        
        return trained_model, evaluation_results, results_df
    
    def generate_optimization_report(self, model, evaluation_results, results_df):
        """Generate comprehensive optimization report"""
        report_path = os.path.join(self.results_dir, 'optimization_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BIO-INSPIRED AIRFOIL OPTIMIZATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results Directory: {self.results_dir}\n\n")
            
            f.write("DATASET SUMMARY:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total bird species analyzed: {len(self.bird_data)}\n")
            f.write(f"Optimization targets selected: {len(self.target_birds)}\n")
            f.write(f"Strategies covered: {len(self.target_birds['optimization_strategy'].unique())}\n\n")
            
            f.write("MORPHOLOGICAL RANGES:\n")
            f.write("-"*40 + "\n")
            morpho_cols = ['Wing.Length', 'Hand-Wing.Index', 'Aspect_Ratio', 'Pointedness_Index']
            for col in morpho_cols:
                if col in self.bird_data.columns:
                    f.write(f"{col}: {self.bird_data[col].min():.1f} - {self.bird_data[col].max():.1f}\n")
            f.write("\n")
            
            f.write("OPTIMIZATION RESULTS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total evaluation episodes: {len(results_df)}\n")
            f.write(f"Mean episode reward: {results_df['total_reward'].mean():.2f} ± {results_df['total_reward'].std():.2f}\n")
            f.write(f"Target achievement rate: {results_df['target_achieved'].mean()*100:.1f}%\n")
            f.write(f"Mean L/D ratio achieved: {results_df['final_cl_cd'].mean():.1f}\n")
            f.write(f"Best L/D ratio achieved: {results_df['final_cl_cd'].max():.1f}\n\n")
            
            f.write("STRATEGY PERFORMANCE:\n")
            f.write("-"*40 + "\n")
            for strategy in results_df['target_strategy'].unique():
                strategy_data = results_df[results_df['target_strategy'] == strategy]
                f.write(f"{strategy.replace('_', ' ').title()}:\n")
                f.write(f"  Episodes: {len(strategy_data)}\n")
                f.write(f"  Mean reward: {strategy_data['total_reward'].mean():.2f}\n")
                f.write(f"  Success rate: {strategy_data['target_achieved'].mean()*100:.1f}%\n")
                f.write(f"  Mean L/D: {strategy_data['final_cl_cd'].mean():.1f}\n\n")
            
            f.write("TOP PERFORMING DESIGNS:\n")
            f.write("-"*40 + "\n")
            top_designs = results_df.nlargest(5, 'total_reward')
            for idx, (_, design) in enumerate(top_designs.iterrows()):
                f.write(f"{idx+1}. Target: {design['target_bird'][:30]}...\n")
                f.write(f"   Strategy: {design['target_strategy']}\n")
                f.write(f"   Reward: {design['total_reward']:.2f}\n")
                f.write(f"   L/D: {design['final_cl_cd']:.1f}\n")
                f.write(f"   CL_max: {design['final_cl_max']:.2f}\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write("-"*40 + "\n")
            f.write("- morphological_analysis.png: Comprehensive bird morphology analysis\n")
            f.write("- optimization_targets.png: Selected optimization targets visualization\n")
            f.write("- training_progress.png: RL training progress and learning curves\n")
            f.write("- evaluation_analysis.png: Detailed evaluation results analysis\n")
            f.write("- optimization_targets.csv: Selected target birds data\n")
            f.write("- evaluation_results.csv: Detailed evaluation results\n")
            f.write("- trained_model_*: Saved RL model files\n\n")
            
            f.write("="*80 + "\n")
        
        logger.info(f"Optimization report saved to {report_path}")


def main():
    """Main execution function for the advanced bio-inspired optimization"""
    
    # Configure paths
    bird_data_path = "./data/AVONET_BIRDLIFE.csv"  # Update with your actual path
    results_dir = "./results/bio_inspired_optimization"
    
    try:
        # Initialize the advanced optimizer
        optimizer = AdvancedBioInspiredOptimizer(
            bird_data_path=bird_data_path,
            results_dir=results_dir
        )
        
        # Run complete optimization pipeline
        trained_model, evaluation_results, results_df = optimizer.run_complete_optimization(
            algorithm='PPO',
            timesteps=50000,  # Reduced for demo, increase for production
            use_xfoil=False,  # Set to True if XFOIL is available
            n_parallel_envs=2  # Adjust based on your system
        )
        
        print("\n🎉 Bio-inspired airfoil optimization completed successfully!")
        print(f"📊 Check results in: {results_dir}")
        
        return optimizer, trained_model, evaluation_results
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Store results globally for notebook access
    global optimizer, model, results
    optimizer, model, results = main()layout()
        plt.savefig(os.path.join(self.results_dir, 'morphological_analysis.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n=== Morphological Analysis Summary ===")
        print(f"Total species analyzed: {len(self.bird_data)}")
        print(f"Wing length range: {self.bird_data['Wing.Length'].min():.1f} - {self.bird_data['Wing.Length'].max():.1f} mm")
        print(f"Aspect ratio range: {self.bird_data['Aspect_Ratio'].min():.1f} - {self.bird_data['Aspect_Ratio'].max():.1f}")
        print(f"Hand-wing index range: {self.bird_data['Hand-Wing.Index'].min():.1f} - {self.bird_data['Hand-Wing.Index'].max():.1f}")
        
    def select_optimization_targets(self, targets_per_category=5):
        """Select diverse optimization targets representing different flight strategies"""
        logger.info("Selecting optimization targets...")
        
        # Define selection criteria for different flight strategies
        selection_criteria = {
            'high_speed': {
                'primary': 'Estimated_Speed_Performance',
                'secondary': 'Hand-Wing.Index',
                'description': 'Optimized for high-speed flight'
            },
            'soaring': {
                'primary': 'Estimated_Soaring_Performance', 
                'secondary': 'Aspect_Ratio',
                'description': 'Optimized for efficient soaring'
            },
            'maneuvering': {
                'primary': 'Estimated_Maneuver_Performance',
                'secondary': 'Secondary_Ratio', 
                'description': 'Optimized for maneuverability'
            },
            'efficient': {
                'primary': 'Hand-Wing.Index',
                'secondary': 'Pointedness_Index',
                'description': 'Optimized for overall efficiency'
            }
        }
        
        self.target_birds = pd.DataFrame()
        
        for strategy, criteria in selection_criteria.items():
            # Select top performers in this category
            category_birds = self.bird_data.nlargest(targets_per_category, criteria['primary'])
            category_birds['optimization_strategy'] = strategy
            category_birds['strategy_description'] = criteria['description']
            
            self.target_birds = pd.concat([self.target_birds, category_birds], ignore_index=True)
        
        logger.info(f"Selected {len(self.target_birds)} optimization targets across {len(selection_criteria)} strategies")
        
        # Save target selection
        self.target_birds.to_csv(os.path.join(self.results_dir, 'optimization_targets.csv'), index=False)
        
        # Visualize target selection
        self.plot_target_selection()
        
        return self.target_birds
    
    def plot_target_selection(self):
        """Visualize selected optimization targets"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Selected Optimization Targets by Strategy', fontsize=16)
        
        strategies = self.target_birds['optimization_strategy'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
        
        # Plot 1: Performance space with targets highlighted
        ax = axes[0, 0]
        
        # Plot all birds as background
        ax.scatter(self.bird_data['Estimated_Speed_Performance'],
                  self.bird_data['Estimated_Soaring_Performance'],
                  c='lightgray', alpha=0.3, s=20, label='All birds')
        
        # Plot targets by strategy
        for strategy, color in zip(strategies, colors):
            strategy_targets = self.target_birds[self.target_birds['optimization_strategy'] == strategy]
            ax.scatter(strategy_targets['Estimated_Speed_Performance'],
                      strategy_targets['Estimated_Soaring_Performance'],
                      c=[color], s=100, alpha=0.8, label=strategy.replace('_', ' ').title(),
                      edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Speed Performance Index')
        ax.set_ylabel('Soaring Performance Index')
        ax.set_title('Performance Space Coverage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Morphological space
        ax = axes[0, 1]
        ax.scatter(self.bird_data['Wing.Length'], self.bird_data['Aspect_Ratio'],
                  c='lightgray', alpha=0.3, s=20)
        
        for strategy, color in zip(strategies, colors):
            strategy_targets = self.target_birds[self.target_birds['optimization_strategy'] == strategy]
            ax.scatter(strategy_targets['Wing.Length'], strategy_targets['Aspect_Ratio'],
                      c=[color], s=100, alpha=0.8, label=strategy.replace('_', ' ').title(),
                      edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Wing Length (mm)')
        ax.set_ylabel('Aspect Ratio')
        ax.set_title('Morphological Space Coverage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Target characteristics radar chart
        ax = axes[1, 0]
        characteristics = ['Wing.Length', 'Hand-Wing.Index', 'Pointedness_Index', 
                          'Secondary_Ratio', 'Aspect_Ratio']
        
        # Normalize characteristics for radar plot
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(self.target_birds[characteristics])
        
        angles = np.linspace(0, 2*np.pi, len(characteristics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), characteristics)
        
        for i, strategy in enumerate(strategies):
            strategy_indices = self.target_birds['optimization_strategy'] == strategy
            strategy_data = normalized_data[strategy_indices].mean(axis=0).tolist()
            strategy_data += strategy_data[:1]  # Complete the circle
            
            ax.plot(angles, strategy_data, 'o-', linewidth=2, label=strategy.replace('_', ' ').title(),
                   color=colors[i], alpha=0.8)
            ax.fill(angles, strategy_data, alpha=0.1, color=colors[i])
        
        ax.set_ylim(0, 1)
        ax.set_title('Strategy Characteristics Profile')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Plot 4: Target distribution by performance
        ax = axes[1, 1]
        performance_metrics = ['Estimated_Speed_Performance', 'Estimated_Soaring_Performance', 
                             'Estimated_Maneuver_Performance']
        
        x = np.arange(len(performance_metrics))
        width = 0.8 / len(strategies)
        
        for i, strategy in enumerate(strategies):
            strategy_targets = self.target_birds[self.target_birds['optimization_strategy'] == strategy]
            means = [strategy_targets[metric].mean() for metric in performance_metrics]
            
            ax.bar(x + i * width, means, width, label=strategy.replace('_', ' ').title(),
                  color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Performance Metrics')
        ax.set_ylabel('Mean Performance Index')
        ax.set_title('Strategy Performance Profiles')
        ax.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax.set_xticklabels([metric.replace('Estimated_', '').replace('_', ' ') for metric in performance_metrics])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_