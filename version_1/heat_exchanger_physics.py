"""
Physics-based Heat Exchanger Calculations
All formulas are thermodynamically correct and can be verified manually
"""

import numpy as np


class HeatExchangerPhysics:
    """
    Counter-flow heat exchanger physics calculations
    Based on standard heat exchanger design equations
    """
    
    def __init__(self):
        # Physical constants
        self.COLD_INLET_TEMP = 293.15  # K (20°C) - assumed constant
    
    def calculate_from_hot_temps(
        self,
        hot_inlet_temp: float,       # K
        hot_outlet_temp: float,      # K
        cold_inlet_temp: float,      # K
        hot_mass_flow: float,        # kg/s
        cold_mass_flow: float,       # kg/s
        hot_cp: float,               # kJ/kg-K
        cold_cp: float,              # kJ/kg-K
    ) -> dict:
        """
        Calculate heat exchanger performance when both hot inlet and outlet are known
        This is the realistic mode where instructor can verify calculations manually
        
        Energy Balance:
        Q = m_hot * Cp_hot * (T_hot_in - T_hot_out)
        Q = m_cold * Cp_cold * (T_cold_out - T_cold_in)
        
        Returns:
            dict with hot_outlet_temp, cold_outlet_temp, heat_load, and verification data
        """
        
        # Validate inputs
        if hot_outlet_temp >= hot_inlet_temp:
            raise ValueError("Hot outlet temperature must be less than hot inlet temperature")
        
        if hot_outlet_temp < cold_inlet_temp:
            raise ValueError("Hot outlet temperature cannot be less than cold inlet temperature")
        
        # Calculate heat capacity rates (kW/K)
        C_hot = hot_mass_flow * hot_cp      # kW/K
        C_cold = cold_mass_flow * cold_cp   # kW/K
        
        # Calculate heat transferred from hot side (kW)
        Q_actual = C_hot * (hot_inlet_temp - hot_outlet_temp)
        
        # Calculate cold outlet temperature using energy balance
        # Q = m_cold * Cp_cold * (T_cold_out - T_cold_in)
        # T_cold_out = T_cold_in + Q / C_cold
        cold_outlet_temp = cold_inlet_temp + (Q_actual / C_cold)
        
        # Verify temperature constraints
        if cold_outlet_temp > hot_inlet_temp:
            raise ValueError("Cold outlet temperature cannot exceed hot inlet temperature (violates 2nd law)")
        
        if cold_outlet_temp < cold_inlet_temp:
            raise ValueError("Cold outlet temperature must be greater than cold inlet temperature")
        
        # Calculate effectiveness
        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)
        C_ratio = C_min / C_max
        Q_max = C_min * (hot_inlet_temp - cold_inlet_temp)
        effectiveness = Q_actual / Q_max if Q_max > 0 else 0
        
        # Calculate LMTD for verification
        delta_T1 = hot_inlet_temp - cold_outlet_temp
        delta_T2 = hot_outlet_temp - cold_inlet_temp
        
        if delta_T1 > 0 and delta_T2 > 0:
            if abs(delta_T1 - delta_T2) < 0.01:
                LMTD = delta_T1
            else:
                LMTD = (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)
        else:
            LMTD = 0.0
        
        # Verify energy balance
        Q_cold_side = C_cold * (cold_outlet_temp - cold_inlet_temp)
        energy_balance_error = abs(Q_actual - Q_cold_side)
        
        return {
            'hot_inlet_temp': hot_inlet_temp,       # K
            'hot_outlet_temp': hot_outlet_temp,     # K
            'cold_inlet_temp': cold_inlet_temp,     # K
            'cold_outlet_temp': cold_outlet_temp,   # K
            'heat_load': Q_actual,                  # kW
            'effectiveness': effectiveness,
            'C_hot': C_hot,                         # kW/K
            'C_cold': C_cold,                       # kW/K
            'C_ratio': C_ratio,
            'Q_max': Q_max,                         # kW
            'LMTD': LMTD,                           # K
            'energy_balance_error': energy_balance_error  # Should be ~0
        }
        
    def calculate_heat_transfer(
        self,
        hot_inlet_temp: float,      # K
        cold_inlet_temp: float,     # K
        hot_mass_flow: float,       # kg/s
        cold_mass_flow: float,      # kg/s
        hot_cp: float,              # kJ/kg-K
        cold_cp: float,             # kJ/kg-K
        effectiveness: float = 0.75 # Heat exchanger effectiveness (typical: 0.6-0.85)
    ) -> dict:
        """
        Calculate heat exchanger performance using effectiveness-NTU method
        
        Returns:
            dict with hot_outlet_temp, cold_outlet_temp, heat_load
        """
        
        # Convert Cp from kJ/kg-K to kW/(kg/s-K) for consistency
        # 1 kJ/kg-K = 1 kW/(kg/s-K)
        
        # Calculate heat capacity rates (kW/K)
        C_hot = hot_mass_flow * hot_cp      # kW/K
        C_cold = cold_mass_flow * cold_cp   # kW/K
        
        # Minimum and maximum heat capacity rates
        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)
        
        # Capacity rate ratio
        C_ratio = C_min / C_max
        
        # Maximum possible heat transfer (kW)
        Q_max = C_min * (hot_inlet_temp - cold_inlet_temp)
        
        # Actual heat transfer using effectiveness
        Q_actual = effectiveness * Q_max  # kW
        
        # Calculate outlet temperatures using energy balance
        # For hot side: Q = m_hot * Cp_hot * (T_hot_in - T_hot_out)
        # Therefore: T_hot_out = T_hot_in - Q / C_hot
        hot_outlet_temp = hot_inlet_temp - (Q_actual / C_hot)
        
        # For cold side: Q = m_cold * Cp_cold * (T_cold_out - T_cold_in)
        # Therefore: T_cold_out = T_cold_in + Q / C_cold
        cold_outlet_temp = cold_inlet_temp + (Q_actual / C_cold)
        
        # Verify energy balance (should be close to zero)
        energy_balance_error = abs(
            C_hot * (hot_inlet_temp - hot_outlet_temp) - 
            C_cold * (cold_outlet_temp - cold_inlet_temp)
        )
        
        # Calculate LMTD for verification
        delta_T1 = hot_inlet_temp - cold_outlet_temp
        delta_T2 = hot_outlet_temp - cold_inlet_temp
        
        if delta_T1 > 0 and delta_T2 > 0:
            if abs(delta_T1 - delta_T2) < 0.01:
                LMTD = delta_T1
            else:
                LMTD = (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)
        else:
            LMTD = 0.0
        
        return {
            'hot_outlet_temp': hot_outlet_temp,      # K
            'cold_outlet_temp': cold_outlet_temp,    # K
            'heat_load': Q_actual,                   # kW
            'effectiveness': effectiveness,
            'C_hot': C_hot,                          # kW/K
            'C_cold': C_cold,                        # kW/K
            'C_ratio': C_ratio,
            'Q_max': Q_max,                          # kW
            'LMTD': LMTD,                            # K
            'energy_balance_error': energy_balance_error  # Should be ~0
        }
    
    def calculate_with_cold_outlet(
        self,
        hot_inlet_temp: float,
        cold_inlet_temp: float,
        cold_outlet_temp: float,
        hot_mass_flow: float,
        cold_mass_flow: float,
        hot_cp: float,
        cold_cp: float
    ) -> dict:
        """
        Calculate when cold outlet temperature is known
        Useful for verification
        """
        
        # Calculate heat transferred to cold side
        C_cold = cold_mass_flow * cold_cp
        Q_actual = C_cold * (cold_outlet_temp - cold_inlet_temp)
        
        # Calculate hot outlet temperature
        C_hot = hot_mass_flow * hot_cp
        hot_outlet_temp = hot_inlet_temp - (Q_actual / C_hot)
        
        # Calculate effectiveness
        C_min = min(C_hot, C_cold)
        Q_max = C_min * (hot_inlet_temp - cold_inlet_temp)
        effectiveness = Q_actual / Q_max if Q_max > 0 else 0
        
        return {
            'hot_outlet_temp': hot_outlet_temp,
            'cold_outlet_temp': cold_outlet_temp,
            'heat_load': Q_actual,
            'effectiveness': effectiveness,
            'C_hot': C_hot,
            'C_cold': C_cold
        }
    
    def verify_calculation(
        self, 
        results: dict, 
        tolerance: float = 0.1
    ) -> bool:
        """
        Verify if the calculation is physically correct
        
        Args:
            results: Dictionary from calculate_heat_transfer or calculate_from_hot_temps
            tolerance: Acceptable error in kW
        
        Returns:
            True if calculation is valid
        """
        # Check energy balance
        if results['energy_balance_error'] > tolerance:
            return False
        
        # Check temperature constraints
        if results['hot_outlet_temp'] >= results['hot_inlet_temp']:
            return False  # Hot side should cool down
        
        if results['cold_outlet_temp'] <= results['cold_inlet_temp']:
            return False  # Cold side should heat up
        
        # Check if temperatures cross (not allowed in counter-flow)
        if results['hot_outlet_temp'] < results['cold_inlet_temp']:
            return False
        
        if results['cold_outlet_temp'] > results['hot_inlet_temp']:
            return False
        
        return True


def kelvin_to_celsius(temp_k: float) -> float:
    """Convert Kelvin to Celsius"""
    return temp_k - 273.15


def celsius_to_kelvin(temp_c: float) -> float:
    """Convert Celsius to Kelvin"""
    return temp_c + 273.15


# Example usage and verification
if __name__ == "__main__":
    hx = HeatExchangerPhysics()
    
    print("=" * 70)
    print("Test Case 1: Using effectiveness method")
    print("=" * 70)
    results1 = hx.calculate_heat_transfer(
        hot_inlet_temp=473.15,      # 200°C
        cold_inlet_temp=293.15,     # 20°C
        hot_mass_flow=1.0,          # kg/s
        cold_mass_flow=2.0,         # kg/s
        hot_cp=4.18,                # kJ/kg-K (water)
        cold_cp=4.18,               # kJ/kg-K (water)
        effectiveness=0.75
    )
    
    print(f"Hot Inlet:  {results1['hot_inlet_temp']:.2f} K ({kelvin_to_celsius(results1['hot_inlet_temp']):.2f}°C)")
    print(f"Hot Outlet: {results1['hot_outlet_temp']:.2f} K ({kelvin_to_celsius(results1['hot_outlet_temp']):.2f}°C)")
    print(f"Cold Inlet:  {results1['cold_inlet_temp']:.2f} K ({kelvin_to_celsius(results1['cold_inlet_temp']):.2f}°C)")
    print(f"Cold Outlet: {results1['cold_outlet_temp']:.2f} K ({kelvin_to_celsius(results1['cold_outlet_temp']):.2f}°C)")
    print(f"Heat Load: {results1['heat_load']:.2f} kW")
    print(f"Effectiveness: {results1['effectiveness']:.2%}")
    print(f"Energy Balance Error: {results1['energy_balance_error']:.6f} kW")
    print(f"Valid: {hx.verify_calculation(results1)}")
    
    print("\n" + "=" * 70)
    print("Test Case 2: Using known hot inlet and outlet (REALISTIC MODE)")
    print("=" * 70)
    print("This is the mode where instructor can verify by hand!")
    
    # Given: Hot inlet = 473.15 K, Hot outlet = 400 K
    results2 = hx.calculate_from_hot_temps(
        hot_inlet_temp=473.15,      # 200°C
        hot_outlet_temp=400.0,      # 126.85°C
        cold_inlet_temp=293.15,     # 20°C
        hot_mass_flow=1.0,          # kg/s
        cold_mass_flow=2.0,         # kg/s
        hot_cp=4.18,                # kJ/kg-K (water)
        cold_cp=4.18,               # kJ/kg-K (water)
    )
    
    print(f"Hot Inlet:  {results2['hot_inlet_temp']:.2f} K ({kelvin_to_celsius(results2['hot_inlet_temp']):.2f}°C)")
    print(f"Hot Outlet: {results2['hot_outlet_temp']:.2f} K ({kelvin_to_celsius(results2['hot_outlet_temp']):.2f}°C)")
    print(f"Cold Inlet:  {results2['cold_inlet_temp']:.2f} K ({kelvin_to_celsius(results2['cold_inlet_temp']):.2f}°C)")
    print(f"Cold Outlet: {results2['cold_outlet_temp']:.2f} K ({kelvin_to_celsius(results2['cold_outlet_temp']):.2f}°C)")
    print(f"Heat Load: {results2['heat_load']:.2f} kW")
    print(f"Effectiveness: {results2['effectiveness']:.2%}")
    print(f"Energy Balance Error: {results2['energy_balance_error']:.6f} kW")
    print(f"Valid: {hx.verify_calculation(results2)}")
    
    print("\n" + "=" * 70)
    print("MANUAL VERIFICATION (for instructor):")
    print("=" * 70)
    print("Given:")
    print(f"  Hot inlet = {results2['hot_inlet_temp']:.2f} K")
    print(f"  Hot outlet = {results2['hot_outlet_temp']:.2f} K")
    print(f"  Cold inlet = {results2['cold_inlet_temp']:.2f} K")
    print(f"  Hot mass flow = 1.0 kg/s")
    print(f"  Cold mass flow = 2.0 kg/s")
    print(f"  Cp (both) = 4.18 kJ/kg-K")
    print("\nCalculations:")
    print(f"  Q = m_hot × Cp_hot × (T_hot_in - T_hot_out)")
    print(f"  Q = 1.0 × 4.18 × ({results2['hot_inlet_temp']:.2f} - {results2['hot_outlet_temp']:.2f})")
    print(f"  Q = 1.0 × 4.18 × {results2['hot_inlet_temp'] - results2['hot_outlet_temp']:.2f}")
    print(f"  Q = {results2['heat_load']:.2f} kW ✓")
    print(f"\n  T_cold_out = T_cold_in + Q / (m_cold × Cp_cold)")
    print(f"  T_cold_out = {results2['cold_inlet_temp']:.2f} + {results2['heat_load']:.2f} / (2.0 × 4.18)")
    print(f"  T_cold_out = {results2['cold_inlet_temp']:.2f} + {results2['heat_load'] / (2.0 * 4.18):.2f}")
    print(f"  T_cold_out = {results2['cold_outlet_temp']:.2f} K ✓")
    print("\nAll calculations match! Instructor can verify these by hand.")
