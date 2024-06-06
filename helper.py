import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prepare_leogo_forecast_hourly(data, day=0, num_days=7):
    """Returns hourly LEOGO forecast data of wind"""
    return np.array(data["curve_wind"][(day) * 24 : 48 * (day + num_days) : 2])


def build_dg_model(
    price_data, wind_data, demand_data, turbine_data, horizon_length, include_grid=False
):
    """Builds Pyomo model of hybrid model

    Args:
        param price_data (list): forecasted price data
        wind_data (list): forecasted wind data
        demand_data (list): demand data
        turbine_data (list): data about the gas turbines
        horizon_length (int): length of the horizon in the pyomo model
        include_grid (bool): Include the grid as a power source
    """

    # Create Concrete Model and corresponding parameters
    m = pyo.ConcreteModel("Hybrid energy model for offshore oil & gas")

    # PARAMETRIC CONTROL OF GAS TURBINES
    device_control_parameters = {"GT1": 1, "GT2": 0, "GT3": 0}

    demand_data = demand_data[:horizon_length]
    wind_data = wind_data[:horizon_length]
    price_data = price_data[:horizon_length]

    # Filters the set of gas turbines based on which ones are online
    generators = ["GT1", "GT2", "GT3"]
    generators = filter(
        lambda x: device_control_parameters[x], device_control_parameters
    )
    gasdata = {"co2_content": 2.34, "energy_value": 40}  # MJ/Sm3

    turbine_data["O&M"] = 0.9  # Flat running cost of operating gas turbines
    co2_tax = 248  # Cost ($) per mwh for released co2 in the atmosphere

    ## Define Sets
    m.top = pyo.RangeSet(0, horizon_length - 1)
    m.N = pyo.Set(
        within=m.top,
        initialize=range(horizon_length),
        doc="Length of prediction horizon",
    )
    m.G = pyo.Set(initialize=generators, doc="All online generators defined by input")
    # --------------------------- Parameters ---------------------------------
    # Square root of round trip efficiency
    m.sqrteta_c = pyo.Param(
        initialize=np.sqrt(0.88),
        mutable=False,
        doc="Square root of round trip efficiency charge",
    )
    m.sqrteta_d = pyo.Param(
        initialize=np.sqrt(0.96),
        mutable=False,
        doc="Square root of round trip efficiency discharge",
    )

    # Parametric device start stop prep off
    m.pDeviceOn = pyo.Param(
        m.G,
        initialize=1,
        domain=pyo.Binary,
        mutable=False,
        doc="On/Off status of generator dev at time t",
    )
    m.pPrice = pyo.Param(
        m.N,
        initialize=price_data,
        domain=pyo.Reals,
        mutable=True,
        doc="Model price over prediction horizon",
    )
    m.pWind = pyo.Param(
        m.N,
        initialize=wind_data,
        domain=pyo.NonNegativeReals,
        mutable=True,
        doc="Model wind forecast over prediction horizon",
    )
    m.pDemand = pyo.Param(
        m.N,
        initialize=demand_data,
        domain=pyo.NonNegativeReals,
        mutable=True,
        doc="Model demand over prediction horizon",
    )
    m.E0 = pyo.Param(
        initialize=0,
        domain=pyo.NonNegativeReals,
        mutable=True,
        doc="Initial bess inventory",
    )

    # ---------------------------- Variables ------------------------------------
    m.P_gt = pyo.Var(
        m.N,
        m.G,
        bounds=lambda model, t, g: (float(turbine_data.flow_min[g]), float(turbine_data.flow_max[g])),
        domain=pyo.NonNegativeReals,
        doc="Power output of each device, with max bound",
    )
    m.vGas = pyo.Var(
        m.N,
        m.G,
        domain=pyo.NonNegativeReals,
        doc="Amount of gas used at time t by device dev",
    )
    m.vChargeRate = pyo.Var(
        m.N, initialize=0.0, bounds=(0.0, 2.0), doc="Charging rate [MW], from 0 - 2"
    )
    m.vDischargeRate = pyo.Var(
        m.N, initialize=0.0, bounds=(0.0, 4.0), doc="Discharge rate [MW], from 0 - 2"
    )
    m.E = pyo.Var(m.N, bounds=(0.0, 6.0), doc="Energy preservation from t, t+T [MW]")
    m.vGrid = pyo.Var(
        m.N,
        domain=pyo.Reals,
        bounds=(-20.0, 20.0),
        doc="Energy bought from the grid at time t, with capacity constraint",
    )
    m.vPenalty = pyo.Var(
        m.N, domain=pyo.Reals, doc="This is just to map out the penalty to the solver"
    )

    # ----------------------------------------------------------------------------
    @m.Expression(m.N, doc="Joint power discharge and charge")
    def P_bess(model, t):
        return model.vChargeRate[t] * model.sqrteta_c - \
                model.vDischargeRate[t] / model.sqrteta_d

    @m.Expression(m.N, doc="Expression, Power generator output")
    def P_generator_output(model, t):
        """Power generated per gas turbine device at time t"""
        return sum(model.P_gt[t, dev] for dev in model.G)

    @m.Expression(m.N, doc="Expression, Power wind output")
    def P_wind_output(model, t):
        """Wind power output, predicted horizon
        Fitting for the wind power
        """
        b, c = 0.07, 0.05
        upsilon = 12 * model.pWind[t]
        return 3 * (c * upsilon**2 + b * upsilon)

    @m.Expression(m.N, doc="Expression, Demand")
    def P_demand(model, t):
        """Demand reqirements, predicted horizon"""
        return model.pDemand[t]

    @m.Constraint(m.N, doc="BESS energy preservation")
    def _rule_battery_preservation(model: pyo.Model, t: int):
        """Battery energy preservation constraint (hourly)"""
        if t == 0:
            return model.E[t] == model.E0
        else:
            return model.E[t] == model.E[t - 1] + model.P_bess[t]

    @m.Constraint(m.N, m.G, doc="GT constraint fuel")
    def _rule_gas_turbine_fuel_usage(model, t, dev):
        """Gas turbine power/gas relation"""
        gas_energy_content = gasdata["energy_value"]
        A = turbine_data.fuel_A[dev]
        B = turbine_data.fuel_B[dev]
        P_max = turbine_data.flow_max[dev]

        lhs = model.vGas[t, dev] * gas_energy_content
        rhs = B * (model.pDeviceOn[dev]) * P_max + A * model.P_gt[t, dev]
        return lhs == rhs

    @m.Constraint(m.N, doc="Power balance constraint")
    def _rule_energy_balance(model, t):
        """
        That energy not spent by the demand goes to battery
        """
        if include_grid:
            return (
                model.pDemand[t]
                == -model.P_bess[t]
                + model.P_generator_output[t]
                + model.P_wind_output[t]
                + model.vGrid[t]
            )
        return (
            model.pDemand[t]
            == -model.P_bess[t] + model.P_generator_output[t] + model.P_wind_output[t]
        )

    # Cost of CO2 is expected to increase
    @m.Expression(doc="Amount of co2 released from running gas turbines")
    def _expr_Cost_co2_released(model):
        return co2_tax * sum(
            model.vGas[t, dev] * gasdata["co2_content"]
            for dev in model.G
            for t in model.N
        )

    @m.Expression(doc="Quadratic cost of running gas turbines")
    def _expr_Cost_quad(model):
        """Represent cost of running as a quadratic equation"""
        a, b, c = 100, 6, 0.005
        return sum(
            c * model.P_gt[t, dev] ** 2 + b * model.P_gt[t, dev] + a
            for t in model.N
            for dev in model.G
        )

    # Quadratic constraint on gas turbine operation
    @m.Expression(doc="Cost of gas turbine operation")
    def _expr_Cost_gas_turbines(model):
        """Cost of running gas turbines"""
        return model._expr_Cost_quad + sum(turbine_data["O&M"][dev] for dev in model.G)

    # Temporary For plotting purpouses
    @m.Constraint(m.N, doc="Net sum of the total penalty at time t")
    def penalty(model, t):
        return (
            model.vPenalty[t]
            >= sum(
                0.005 * model.P_gt[t, dev] ** 2 + 6 * model.P_gt[t, dev] + 100
                for dev in model.G
            )
            + co2_tax
            * sum(model.vGas[t, dev] * gasdata["co2_content"] for dev in model.G)
            - 100 * (model.vDischargeRate[t] - model.vChargeRate[t])
            + model.pPrice[t] * model.vGrid[t]
        )

    @m.Objective(sense=pyo.minimize)
    def min_objective(model):
        """Minimizes the total penalty on the system at each time t"""
        return sum(model.vPenalty[t] for t in model.N)

    return m


# ----------------------------------------------------------------------------------------------------------------------------------------------


def build_dg_model_MPC(
    price_data, wind_data, demand_data, turbine_data, horizon_length, include_grid=False
):
    """Hybridmodel for mpc implementation

    Args:
        param price_data (list): forecasted price data
        wind_data (list): forecasted wind data
        demand_data (list): demand data
        turbine_data (list): data about the gas turbines
        horizon_length (int): length of the horizon in the pyomo model
        include_grid (bool): Include the grid as a power source
    """

    # Create Concrete Model and corresponding parameters
    m = pyo.ConcreteModel("MPC hybrid energy model")

    # CONTROL PARAMETERS FOR SYSTEM
    device_control_parameters = {"GT1": 1, "GT2": 0, "GT3": 0}
    demand_data = demand_data[:horizon_length]
    wind_data = wind_data[:horizon_length]
    price_data = price_data[:horizon_length]

    # Filters the set of gas turbines based on input
    generators = ["GT1", "GT2", "GT3"]
    generators = filter(
        lambda x: device_control_parameters[x], device_control_parameters
    )

    gasdata = {
        "co2_content": 2.34,
        "energy_value": 40,
    }
    turbine_data["O&M"] = 0.9  # Flat running cost of operating gas turbines

    co2_tax = 248  # Cost ($) per mwh for released co2 in the atmosphere

    ## Define Sets
    m.top = pyo.RangeSet(0, horizon_length - 1)
    m.N = pyo.Set(
        within=m.top,
        initialize=range(horizon_length),
        doc="Length of prediction horizon",
    )
    m.G = pyo.Set(initialize=generators, doc="All online generators defined by input")

    # --------------------------- Parameters ---------------------------------
    # Square root of round trip efficiency
    m.sqrteta_c = pyo.Param(
        initialize=np.sqrt(0.88),
        mutable=False,
        doc="Square root of round trip efficiency charge",
    )
    m.sqrteta_d = pyo.Param(
        initialize=np.sqrt(0.96),
        mutable=False,
        doc="Square root of round trip efficiency discharge",
    )
    # Parametric device start stop prep off
    m.pDeviceOn = pyo.Param(
        m.G,
        initialize=1,
        domain=pyo.Binary,
        mutable=False,
        doc="On/Off status of generator dev at time t",
    )
    m.pPrice = pyo.Param(
        m.N,
        initialize=price_data[:horizon_length],
        domain=pyo.Reals,
        mutable=True,
        doc="Model price over prediction horizon",
    )
    m.pWind = pyo.Param(
        m.N,
        initialize=wind_data[:horizon_length],
        domain=pyo.NonNegativeReals,
        mutable=True,
        doc="Model wind forecast over prediction horizon",
    )
    m.pDemand = pyo.Param(
        m.N,
        initialize=demand_data,
        domain=pyo.NonNegativeReals,
        mutable=True,
        doc="Model demand over prediction horizon",
    )
    m.E0 = pyo.Param(
        initialize=0,
        domain=pyo.NonNegativeReals,
        mutable=True,
        doc="Initial bess inventory",
    )

    # ---------------------------- Variables ------------------------------------
    m.P_gt = pyo.Var(
        m.N,
        m.G,
        bounds=lambda model, t, g: (float(turbine_data.flow_min[g]), float(turbine_data.flow_max[g])),
        domain=pyo.NonNegativeReals,
        doc="Power output of each device, with max bound",
    )
    m.vGas = pyo.Var(
        m.N,
        m.G,
        domain=pyo.NonNegativeReals,
        doc="Amount of gas used at time t by device dev",
    )
    m.vChargeRate = pyo.Var(
        m.N, initialize=0.0, bounds=(0.0, 2.0), doc="Charging rate [MW], from 0 - 2"
    )
    m.vDischargeRate = pyo.Var(
        m.N, initialize=0.0, bounds=(0.0, 4.0), doc="Discharge rate [MW], from 0 - 2"
    )
    m.E = pyo.Var(m.N, bounds=(0.0, 6.0), doc="Energy preservation from t, t+T [MW]")
    m.vGrid = pyo.Var(
        m.N,
        domain=pyo.Reals,
        bounds=(-20.0, 20.0),
        doc="Energy bought from the grid at time t, with capacity constraint",
    )
    m.vPenalty = pyo.Var(
        m.N, domain=pyo.Reals, doc="This is just to map out the penalty to the solver"
    )
    # ----------------------------------------------------------------------------

    @m.Expression(m.N, doc="Joint power discharge and charge")
    def P_bess(model, t):
        return model.vChargeRate[t] * model.sqrteta_c - \
            model.vDischargeRate[t] / model.sqrteta_d
        

    @m.Expression(m.N, doc="Expression, Power generator output")
    def P_generator_output(model, t):
        """Power generated per gas turbine device at time t"""
        return sum(model.P_gt[t, dev] for dev in model.G)

    @m.Expression(m.N, doc="Expression, Power wind output")
    def P_wind_output(model, t):
        """Wind power output, predicted horizon
        Fitting for the wind power
        """
        b, c = 0.07, 0.05
        upsilon = 12 * model.pWind[t]
        return 3 * (c * upsilon**2 + b * upsilon)

    @m.Expression(m.N, doc="Expression, Demand")
    def P_demand(model, t):
        """Demand reqirements, predicted horizon"""
        return model.pDemand[t]

    @m.Constraint(m.N, doc="BESS energy preservation")
    def _rule_battery_preservation(model: pyo.Model, t: int):
        """
        Battery energy preservation constraint (hourly)
        """
        # First timestep
        if t == 0:
            return model.E[t] == model.E0
        else:
            return model.E[t] == model.E[t - 1] + model.P_bess[t]

    @m.Constraint(m.N, m.G, doc="GT constraint fuel")
    def _rule_gas_turbine_fuel_usage(model, t, dev):
        """Gas turbine power/gas relation"""
        gas_energy_content = gasdata["energy_value"]
        A = turbine_data.fuel_A[dev]
        B = turbine_data.fuel_B[dev]
        P_max = turbine_data.flow_max[dev]

        lhs = model.vGas[t, dev] * gas_energy_content
        rhs = B * (model.pDeviceOn[dev]) * P_max + A * model.P_gt[t, dev]
        return lhs == rhs

    @m.Constraint(m.N, doc="Power balance constraint")
    def _rule_energy_balance(model, t):
        """
        That energy not spent by the demand goes to battery
        MPC controls the initial state, so all variables are frozen in this model
        """
        if t == 0:
            return pyo.Constraint.Skip
        if include_grid:
            return (
                model.pDemand[t]
                == -model.P_bess[t]
                + model.P_generator_output[t]
                + model.P_wind_output[t]
                + model.vGrid[t]
            )
        return (
            model.pDemand[t]
            == -model.P_bess[t] + model.P_generator_output[t] + model.P_wind_output[t]
        )

    # Cost of CO2 is expected to increase
    @m.Expression(doc="Amount of co2 released from running gas turbines")
    def _expr_Cost_co2_released(model):
        return co2_tax * sum(
            model.vGas[t, dev] * gasdata["co2_content"]
            for dev in model.G
            for t in model.N
        )

    @m.Expression(doc="Quadratic cost of running gas turbines")
    def _expr_Cost_quad(model):
        """Represent cost of running as a quadratic equation"""
        a, b, c = 100, 6, 0.005
        return sum(
            c * model.P_gt[t, dev] ** 2 + b * model.P_gt[t, dev] + a
            for t in model.N
            for dev in model.G
        )

    # Quadratic constraint on gas turbine operation
    @m.Expression(doc="Cost of gas turbine operation")
    def _expr_Cost_gas_turbines(model):
        """Cost of running gas turbines"""
        return model._expr_Cost_quad + sum(turbine_data["O&M"][dev] for dev in model.G)

    # Temporary For plotting purpouses
    @m.Constraint(m.N, doc="Net sum of the total penalty at time t")
    def penalty(model, t):
        return (
            model.vPenalty[t]
            >= sum(
                0.005 * model.P_gt[t, dev] ** 2
                + 6 * model.P_gt[t, dev]
                + 100
                + turbine_data["O&M"][dev]
                for dev in model.G
            )
            + co2_tax
            * sum(model.vGas[t, dev] * gasdata["co2_content"] for dev in model.G)
            - 100 * (model.vDischargeRate[t] - model.vChargeRate[t])
            + model.pPrice[t] * model.vGrid[t]
        )

    @m.Objective(sense=pyo.minimize)
    def min_objective(model):
        """Minimizes the total penalty on the system at each time t"""
        return sum(model.vPenalty[t] for t in model.N)

    return m


#################################### MPC ############################################
def run_rhc3(
    m: pyo.Model,
    horizon_length: int,
    f_data: dict,
    n_data: dict,
    sim_steps: int,
    initial_point: mpc.ScalarData,
    nowcast_horizon: int = 2,
    verbose=False,
):
    """Run EMPC for energy model"""

    assert (
        sim_steps >= horizon_length
    ), "Number of simulations must be greater than the length of the horizon!"
    model_interface = mpc.DynamicModelInterface(m, m.N)

    # The range is set to a finite value within the simulation steps
    forecast_data = mpc.TimeSeriesData(f_data, range(sim_steps))
    nowcast_data = mpc.TimeSeriesData(n_data, range(sim_steps))

    # Model interface is the data loaded in the model
    model_interface.load_data(initial_point)  # Loads to all points
    m.E0 = initial_point.get_data_from_key("E[*]")

    # The steady state of the first choice
    sim_data = model_interface.get_data_at_time([0])
    solver = pyo.SolverFactory("gurobi")

    # Simulation steps
    for sim_t0 in range(sim_steps - horizon_length):

        if verbose:
            print(f"Solving for sim t0: {sim_t0}")

        # Set the simulation horizon based on prediction horizon length
        sim_time = [sim_t0 + t for t in m.N]
        hour_6_horizon = [sim_t0 + t for t in range(1, nowcast_horizon)]

        new_forecast = forecast_data.get_data_at_time(sim_time)
        new_forecast.shift_time_points(-sim_t0)
        # Now the new data should index from 0, but 6 and outwards
        model_interface.load_data(
            new_forecast, time_points=new_forecast.get_time_points()
        )

        if sim_t0 % nowcast_horizon == 0:
            new_nowcast = nowcast_data.get_data_at_time(
                hour_6_horizon
            )  # this is the length of the loaded
            new_nowcast.shift_time_points(-sim_t0)
            model_interface.load_data(
                new_nowcast, time_points=new_nowcast.get_time_points()
            )
        else:
            moving = hour_6_horizon[: nowcast_horizon - sim_t0 % nowcast_horizon]
            new_nowcast = nowcast_data.get_data_at_time(moving)
            new_nowcast.shift_time_points(-sim_t0)
            # Load data from 6 hours and onward
            model_interface.load_data(
                new_nowcast, time_points=new_nowcast.get_time_points()
            )

        if verbose:
            df = storage_extract_singleindex(m)
            print(df["pPrice"].tolist())

        # Solve model to simulate the optimal trajectory over the prediction horzion
        res = solver.solve(m)
        pyo.assert_optimal_termination(res)
        m_data = model_interface.get_data_at_time([1])

        # Shift so model time matches the simulation time m.N.first is always 0
        m_data.shift_time_points(sim_t0)

        # Add extracted next optimal step to the simulation modl
        sim_data.concatenate(m_data)

        # Load the new initial data to the model for receding horizon
        tf_data = model_interface.get_data_at_time(1)
        model_interface.load_data(tf_data)
        m.E0 = tf_data.get_data_from_key("E[*]")

    # The horizon has cached up to its stage
    # Forecasting data is stopped
    model_t0 = 0
    for sim_t0 in range(sim_steps - horizon_length, sim_steps - 1):
        hour_6_horizon = [
            sim_t0 + t for t in range(1, min(nowcast_horizon, sim_steps - sim_t0))
        ]

        if sim_t0 % nowcast_horizon == 0:
            new_nowcast = nowcast_data.get_data_at_time(
                hour_6_horizon
            )  # this is the length of the loaded
            new_nowcast.shift_time_points(-sim_t0 + model_t0)
            model_interface.load_data(
                new_nowcast, time_points=new_nowcast.get_time_points()
            )
        else:
            moving = hour_6_horizon[: (nowcast_horizon - sim_t0) % nowcast_horizon]
            new_nowcast = nowcast_data.get_data_at_time(moving)
            new_nowcast.shift_time_points(-sim_t0 + model_t0)
            # Load data from 6 hours and onward
            model_interface.load_data(
                new_nowcast, time_points=new_nowcast.get_time_points()
            )

        if verbose:
            print(new_nowcast.to_serializable())
            df = storage_extract_singleindex(m)
            print(df["pPrice"].tolist())

        # Solve model to simulate the optimal trajectory over the prediction horzion
        # Fix variable as we are approaching the final sim steps
        m.E[model_t0].fix()
        res = solver.solve(m, logfile="if-infeasible-basic.log")
        pyo.assert_optimal_termination(res)
        # Get next control actions
        model_t0 += 1
        # Extract optimal next step, or the final solve if simulation is finished
        m_data = model_interface.get_data_at_time([model_t0])
        m_data.shift_time_points(sim_t0 - model_t0 + 1)
        sim_data.concatenate(m_data)

    print(sim_data.to_serializable())
    return m, sim_data


def run_hybrid_rhc3(
    m: pyo.Model,
    horizon_length: int,
    f_data: dict,
    n_data: dict,
    sim_steps: int,
    initial_point: mpc.ScalarData,
    nowcast_horizon: int = 2,
    verbose=False,
):
    """Run MPC for hybrid model"""

    assert (
        sim_steps >= horizon_length
    ), "Number of simulations must be greater than the length of the horizon!"

    model_interface = mpc.DynamicModelInterface(m, m.N)

    # The range is set to a finite value within the simulation steps
    forecast_data = mpc.TimeSeriesData(f_data, range(sim_steps))
    nowcast_data = mpc.TimeSeriesData(n_data, range(sim_steps))

    # Model interface is the data loaded in the model
    model_interface.load_data(initial_point)  # Loads to all points

    # The steady state of the first choice
    sim_data = model_interface.get_data_at_time([0])
    solver = pyo.SolverFactory("gurobi")

    # Need to set the initial
    m.E0 = initial_point.get_data_from_key("E[*]")
    # Simulation steps
    for sim_t0 in range(sim_steps - horizon_length):
        if verbose:
            print(f"Solving for sim t0: {sim_t0}")

        sim_time = [sim_t0 + t for t in m.N]
        # Set the simulation horizon based on prediction horizon length
        hour_6_horizon = [sim_t0 + t for t in range(1, nowcast_horizon)]

        new_forecast = forecast_data.get_data_at_time(sim_time)
        new_forecast.shift_time_points(-sim_t0)
        # Now the new data should index from 0, but 6 and outwards
        model_interface.load_data(
            new_forecast, time_points=new_forecast.get_time_points()
        )

        if sim_t0 % nowcast_horizon == 0:
            new_nowcast = nowcast_data.get_data_at_time(
                hour_6_horizon
            )  # this is the length of the loaded
            new_nowcast.shift_time_points(-sim_t0)
            model_interface.load_data(
                new_nowcast, time_points=new_nowcast.get_time_points()
            )
        else:
            moving = hour_6_horizon[: (nowcast_horizon - sim_t0 % nowcast_horizon)]
            new_nowcast = nowcast_data.get_data_at_time(moving)
            new_nowcast.shift_time_points(-sim_t0)
            # Load data from 6 hours and onward
            model_interface.load_data(
                new_nowcast, time_points=new_nowcast.get_time_points()
            )

        if verbose:
            df = hybrid_get_deterministic_dataframe(m)
            print(f"pWind: {df['pWind'].tolist()}")

        # Solve model to simulate the optimal trajectory over the prediction horzion
        # Fix Battery SoC for E[0] so that the model uses the initial state
        m.P_gt[0, :].fix()
        m.vGrid[0].fix()
        res = solver.solve(m)  
        pyo.assert_optimal_termination(res)

        # Unfix to load new data
        m.vPenalty[0].unfix()
        m.P_gt[0, :].unfix()
        m.vGrid[0].unfix()

        m_data = model_interface.get_data_at_time([1])
        m_data.shift_time_points(sim_t0)

        # Add extracted next optimal step to the simulation modl
        sim_data.concatenate(m_data)
        # Doing an operation, and getting nothing in return
        tf_data = model_interface.get_data_at_time(1)

        model_interface.load_data(tf_data, time_points=[0])
        m.E0 = tf_data.get_data_from_key("E[*]")

    # The horizon has cached up to its stage
    model_t0 = 0
    for sim_t0 in range(sim_steps - horizon_length, sim_steps - 1):
        hour_6_horizon = [
            sim_t0 + t for t in range(1, min(nowcast_horizon, sim_steps - sim_t0))
        ]

        # Set the simulation horizon based on prediction horizon length
        if sim_t0 % nowcast_horizon == 0:
            new_nowcast = nowcast_data.get_data_at_time(
                hour_6_horizon
            )  # this is the length of the loaded
            new_nowcast.shift_time_points(-sim_t0 + model_t0)
            model_interface.load_data(
                new_nowcast, time_points=new_nowcast.get_time_points()
            )
        else:
            moving = hour_6_horizon[: (nowcast_horizon - sim_t0) % nowcast_horizon]
            new_nowcast = nowcast_data.get_data_at_time(moving)
            new_nowcast.shift_time_points(-sim_t0 + model_t0)
            # Load data from 6 hours and onward
            model_interface.load_data(
                new_nowcast, time_points=new_nowcast.get_time_points()
            )

        if verbose:
            df = hybrid_get_deterministic_dataframe(m)
            print(f"pWind: {df['pWind'].tolist()}")

        # Fix variable as we are approaching the final sim steps
        res = solver.solve(m, logfile="if-infeasible-basic.log")
        # log_infeasible_constraints(m, log_expression=True, log_variables=True)
        pyo.assert_optimal_termination(res)

        # Get next control actions
        model_t0 += 1
        m.P_gt[model_t0, :].fix()
        m.vGrid[model_t0].fix()
        m.E[model_t0].fix()
        m.vDischargeRate[model_t0].fix()
        m.vChargeRate[model_t0].fix()

        # Extract optimal next step, or the final solve if simulation is finished
        m_data = model_interface.get_data_at_time([model_t0])
        m_data.shift_time_points(sim_t0 - model_t0 + 1)
        sim_data.concatenate(m_data)

    print(sim_data.to_serializable())
    return m, sim_data


def storage_extract_singleindex(m):
    """Extract the energy-models variables and parameters"""
    df = pd.DataFrame(
        {
            "vChargeRate": [m.vChargeRate[i].value for i in m.N],
            "vDischargeRate": [m.vDischargeRate[i].value for i in m.N],
            "E": [m.E[i].value for i in m.N],
            "pPrice": [m.pPrice[i].value for i in m.N],
            "netsum": [m.netsum[i]() for i in m.N],
            "N": [i for i in m.N],
        }
    )
    df.set_index("N", inplace=True)
    df["vDischargeRate"] *= -1
    df["Cumulative Profit"] = df["netsum"].cumsum()
    return df


def hybrid_get_deterministic_dataframe(m: pyo.Model):
    """Extract the deterministic hybridmodel variables and parameters"""
    df = pd.DataFrame(
        {
            "vChargeRate": [m.vChargeRate[i].value for i in m.N],
            "vGrid": [m.vGrid[i].value for i in m.N],
            "pDemand": [m.pDemand[i].value for i in m.N],
            "P_wind_output": [m.P_wind_output[i]() for i in m.N],
            "vDischargeRate": [m.vDischargeRate[i].value for i in m.N],
            "E": [m.E[i].value for i in m.N],
            "vPenalty": [m.vPenalty[i].value for i in m.N],
            "P_gt": [m.P_gt[i, j].value for j in m.G for i in m.N],
            "vGas": [m.vGas[i, j].value for j in m.G for i in m.N],
            "pWind": [m.pWind[i].value for i in m.N],
            "N": [i for i in m.N],
        }
    )
    df.set_index("N", inplace=True)
    df["vDischargeRate"] *= -1
    df["Cumulative Penalty"] = df["vPenalty"].cumsum()
    df["CO2"] = 2.34 * df["vGas"]
    return df


def calculate_hybridmodel_actualcost(df_res, price):
    """Calculates the actual cost from the forecasted variables on actual historic data"""
    data = {
        "P_gt": df_res["P_gt"],
        "CO2": df_res["CO2"],
        "vDischargeRate": df_res["vDischargeRate"],
        "vChargeRate": df_res["vChargeRate"],
        "pPrice": price,
        "vGrid": df_res["vGrid"],
    }

    df = pd.DataFrame(data)
    df["turbine_cost"] = 0.005 * df["P_gt"] ** 2 + 6 * df["P_gt"] + 100 + 0.9
    df["co2_cost"] = 248 * df["CO2"]
    df["battery_operation"] = 100 * (df["vDischargeRate"] - df["vChargeRate"])
    df["grid_cost"] = df["pPrice"] * df["vGrid"]
    df["vPenalty"] = (
        df["turbine_cost"] + df["co2_cost"] - df["battery_operation"] + df["grid_cost"]
    )
    df["Cumulative Penalty"] = df["vPenalty"].cumsum()
    return df


def generate_variable_demand(days=7):
    """Generates variable demand with peak of 40 MWh and normal of 30 MWh"""
    np.random.seed(42)
    total = 24 * days
    normal = 30
    peak = 40
    base = np.full(total, normal)

    for day in range(days):
        peak_hours = np.random.choice(
            range(day * 24 + 17, day * 24 + 22), 2, replace=False
        )
        base[peak_hours] = peak
    noise = base + np.random.normal(0, 2, total)

    plt.figure(figsize=(11, 4))
    plt.plot(noise, label="Hourly Demand", color="blue")
    plt.xlabel("Hour")
    plt.ylabel("Demand (MWh)")
    plt.show()
    return noise
