# Model Predictive Control for Hybrid Energy Systems

**Author:** Sondre Ekkje Kaavik  
**Date:** 04.06.24  
**Institution:** University of Bergen, Departement of Informatics   

### Description
The repository contains the energy system for Chapter 4 and the hybrid energy system from Chapter 5.  

### Requirements
List the prerequisites needed to run these notebooks:
- **Python Version:** Implemented for Python 3.12  
- **Libraries and Dependencies:** Including these dependencies, one needs a valid Gurobi license or swap solver. The thesis was solved with Gurobi.  
  ```bash
  pip install numpy pandas matplotlib plotly pyomo gurobipy scipy requests
  ```

### Installation
Jupyter-notebook is required to run the files.  

### File Descriptions
- `data/`: The various data files from LEOGO and Oogeso.  
- `helper.py`: Various methods for plotting and models. Imported in the other files.  
- `energy_storage_model.ipynb`: The EMPC implementation with plotting data from chapter 4.  
- `hybrid_energy_model.ipynb`: The hybrid energy model from chapter 5.  
- Add additional notebooks as necessary.  

### Usage
To run the notebooks, write this in your terminal of choice.
```bash
jupyter notebook energy_storage_model.ipynb
```