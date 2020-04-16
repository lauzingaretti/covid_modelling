# jit just in time 
import numpy as np 
import pandas as pd
import pymc3 as pm 
from scipy.integrate import solve_ivp 
from scipy import optimize 
from numba import jit
import theano #to wrapper
import theano.tensor as t 
import matplotlib.pyplot as plt
import altair as alt

@jit(nopython=True)
def sir_model(t, X, beta=1, gamma=1/15):
    S, I, R = X
    S_prime = - beta * S * I
    I_prime = beta * S * I - gamma * I
    R_prime = gamma * I
    return S_prime, I_prime, R_prime


@jit(nopython=True)
def sirq_model(t, X, beta=1, omega=0,gamma=1/15):
    S,I, R = X
    S_prime = - beta * S * I - omega * S
    I_prime = beta * S * I - gamma * I
    R_prime = gamma * I  
    return  S_prime, I_prime, R_prime

# sird add 1 equation to take into account the death 
@jit(nopython=True)
def sird_model(t, X, beta=1, delta=0.02, gamma=1/15):
    """
    SIR model that takes into account the number of deaths.
    """
    S, I, R, D = X
    S_prime = - beta * S * I
    I_prime = beta * S * I - gamma * I - delta * I
    R_prime = gamma * I
    D_prime = delta * I
    return S_prime, I_prime, R_prime, D_prime

@jit(nopython=True)

def sirdq_model(t, X, beta=1, delta=0.02, omega=0, gamma=1/15):
    """
    SIR model that takes into account the number of deaths.
    """
    S, I, R, D = X
    S_prime = - beta * S * I - omega * S
    I_prime = beta * S * I - gamma * I - delta * I
    R_prime = gamma * I
    D_prime = delta * I
    return S_prime, I_prime, R_prime, D_prime




@jit(nopython=True)
def seir_model(t, X, beta=1, psi=0, gamma=1/14,alpha=1/5):
    """
    This is a modified SEIR model in order to take into account incubation time in exposed individual.
    The exposed individuals can transmit the infection to susceptible individuals.
    """
    S, E, I, R = X
    S_prime = - beta * S * I - psi* E * S
    E_prime = beta * S * I - alpha * E + psi * E * S
    I_prime = alpha * E -gamma * I 
    R_prime = gamma* I
    return S_prime, E_prime, I_prime, R_prime


@jit(nopython=True)
def seirdq_model(t, X,  beta=1, psi=0, delta=0.02,omega=0,alpha=1/5,gamma=1/14):
    """
    A modified SEIR model in order to take into account deaths.
    """
    S, E, I, R, D = X
    S_prime = - beta * S * I - psi * E * S - omega * S
    E_prime = beta * S * I - alpha * E + psi * E * S - omega * E
    I_prime = alpha * E - gamma * I - delta * I - omega * E
    R_prime = gamma * I + omega * (S+E+I)
    D_prime = delta * I
    return S_prime, E_prime, I_prime, R_prime, D_prime




def sir_ode_solver(y0, t_span, t_eval, beta=1, gamma=1/14):
    solution_ODE = solve_ivp(
        fun=lambda t, y: sir_model(t, y, beta=beta, gamma=gamma), 
        t_span=t_span, 
        y0=y0,
        t_eval=t_eval,
        method='LSODA')
    
    return solution_ODE

def sirq_ode_solver(y0, t_span, t_eval, beta=1, omega =0,gamma=1/14):
    solution_ODE = solve_ivp(
        fun=lambda t, y: sirq_model(t, y, beta=beta,omega=omega,gamma=gamma), 
        t_span=t_span, 
        y0=y0,
        t_eval=t_eval,
        method='LSODA')
    
    return solution_ODE

def sird_ode_solver(y0, t_span, t_eval, beta=1, delta=0.02, gamma=1/14):
    solution_ODE = solve_ivp(
        fun=lambda t, y: sird_model(t, y, beta=beta, gamma=gamma, delta=delta), 
        t_span=t_span, 
        y0=y0,
        t_eval=t_eval,
        method='LSODA'
    )
    
    return solution_ODE

def sirdq_ode_solver(y0, t_span, t_eval, beta=1, delta=0.02, omega=0 ,gamma=1/14):
    solution_ODE = solve_ivp(
        fun=lambda t, y: sirdq_model(t, y, beta=beta, delta=delta,omega=omega,gamma=gamma), 
        t_span=t_span, 
        y0=y0,
        t_eval=t_eval,
        method='LSODA'
    )
    
    return solution_ODE

def seir_ode_solver(y0, t_span, t_eval, beta=1, psi=0, alpha=1/5,gamma=1/15):
    solution_ODE = solve_ivp(
        fun=lambda t, y: seir_model(t, y,  beta=beta, psi=psi, alpha=alpha,gamma=gamma), 
        t_span=t_span, 
        y0=y0,
        t_eval=t_eval,
        method='LSODA')
    
    return solution_ODE




def seirdq_ode_solver(y0, t_span, t_eval, beta=1, psi=0, delta=0.02, omega=0,alpha=1/5,gamma=1/15):
    solution_ODE = solve_ivp(
        fun=lambda t, y: seirdq_model(t, y, beta=beta, psi=psi, delta=delta,omega=omega,alpha=alpha,gamma=gamma), 
        t_span=t_span, 
        y0=y0,
        t_eval=t_eval, 
        method='LSODA')

    return solution_ODE



def sir_least_squares_error_ode(par, time_exp, f_exp, fitting_model, initial_conditions):
    #parameters
    args = par
    #cuanto tiempo de epidemia
    time_span = (time_exp.min(), time_exp.max())
    #fitting model es para ajustar el modelo, requiere condiciones iniciales, periodo de tiempo y args
    y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    _, simulated_qoi, _ = simulated_ode_solution
    
    #f exp final 
    residual = f_exp - simulated_qoi

    return np.sum(residual ** 2.0)

def sirq_least_squares_error_ode(par, time_exp, f_exp, fitting_model, initial_conditions):
    #parameters
    args = par
    #cuanto tiempo de epidemia
    time_span = (time_exp.min(), time_exp.max())
    #fitting model es para ajustar el modelo, requiere condiciones iniciales, periodo de tiempo y args
    y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    _, simulated_qoi, _ = simulated_ode_solution
    
    #f exp final 
    residual = f_exp - simulated_qoi

    return np.sum(residual ** 2.0)

def sird_least_squares_error_ode(par, time_exp, f_exp, fitting_model, initial_conditions):
    args = par
    f_exp1, f_exp2 = f_exp
    time_span = (time_exp.min(), time_exp.max())
    
    y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    _, simulated_qoi1, _, simulated_qoi2 = simulated_ode_solution
    
    residual1 = f_exp1 - simulated_qoi1
    residual2 = f_exp2 - simulated_qoi2

    weighting_for_exp1_constraints = 1e0
    weighting_for_exp2_constraints = 1e0
    
    return weighting_for_exp1_constraints * np.sum(residual1 ** 2.0) + weighting_for_exp2_constraints * np.sum(residual2 ** 2.0)


def sirdq_least_squares_error_ode(par, time_exp, f_exp, fitting_model, initial_conditions):
    args = par
    f_exp1, f_exp2 = f_exp
    time_span = (time_exp.min(), time_exp.max())
    
    y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    _, simulated_qoi1, _, simulated_qoi2 = simulated_ode_solution
    
    residual1 = f_exp1 - simulated_qoi1
    residual2 = f_exp2 - simulated_qoi2

    weighting_for_exp1_constraints = 1e0
    weighting_for_exp2_constraints = 1e0
    
    return weighting_for_exp1_constraints * np.sum(residual1 ** 2.0) + weighting_for_exp2_constraints * np.sum(residual2 ** 2.0)


def seir_least_squares_error_ode(par, time_exp, f_exp, fitting_model, initial_conditions):
    args = par
    time_span = (time_exp.min(), time_exp.max())
    
    y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    _, _, simulated_qoi, _ = simulated_ode_solution
    
    residual = f_exp - simulated_qoi

    return np.sum(residual ** 2.0)


def seirdq_least_squares_error_ode(par, time_exp, f_exp, fitting_model, initial_conditions):
    args = par
    f_exp1, f_exp2 = f_exp
    time_span = (time_exp.min(), time_exp.max())
    
    y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
    simulated_time = y_model.t
    simulated_ode_solution = y_model.y
    _, _, simulated_qoi1, _, simulated_qoi2 = simulated_ode_solution
    
    residual1 = f_exp1 - simulated_qoi1
    residual2 = f_exp2 - simulated_qoi2

    weighting_for_exp1_constraints = 1e0
    weighting_for_exp2_constraints = 1e0
    return weighting_for_exp1_constraints * np.sum(residual1 ** 2.0) + weighting_for_exp2_constraints * np.sum(residual2 ** 2.0)


def callback_de(xk, convergence):
    print(f'parameters = {xk}')