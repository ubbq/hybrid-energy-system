# Model Predictive Control for Hybrid Energy Systems

**Author:** Sondre Ekkje Kaavik  
**Date:** 04.06.24  
**Institution:** University of Bergen, Department of Informatics   

### Description
The repository contains the energy system model created for Chapter 4 and the hybrid energy system model from Chapter 5. 

### Requirements
The thesis was completed using Python 3.12, but older versions should also be supported.
- **Libraries and Dependencies:** A valid Gurobi license is needed to use the Gurobi solver. We used Gurobi for the backend in Pyomo.  
  ```bash
  pip install -r requirements
  ```
### File Descriptions
- `data/`: Data files from LEOGO, Oogeso and OpenData.  
- `helper.py`: Models and plotting methods. Also contains the MPC implementations. 
- `energy_storage_model.ipynb`: To run the energy model from Chapter 4.  
- `hybrid_energy_model.ipynb`: To run the hybrid energy model from Chapter 5.  