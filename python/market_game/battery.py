from dataclasses import dataclass

BATTERY_CAPCITY = 20.0  # Maximum battery capacity in kWh

BATTERY_MAX_DISCHARGE = 10.0  # Maximum discharge rate in kWh
BATTERY_MAX_CHARGE = 5.0  # Maximum charge rate in kWh

@dataclass
class Battery:
    energy: float=0
    
    def current_charge(self)->float:
        return self.energy
    
    def discharge(self,delta:float)->float:
        """ discharge the battery by a given value
        assumed to be in 1 hour
        """
        delta=abs(delta)
        if delta>self.energy:
            raise ValueError("requested discharge exceeds current charge level")
        if delta>BATTERY_MAX_DISCHARGE:
            raise ValueError("requested discharge exceeds maximum discharge rate (10)")
        self.energy-=delta
        return self.energy
    
    def charge(self,delta:float)->float:
        """ charge the battery by a given value
        assumed to be in 1 hour
        """
        delta=abs(delta)
        if self.energy+delta>BATTERY_CAPCITY:
            raise ValueError("requested charge exceeds maximum capacity")
        if delta>BATTERY_MAX_CHARGE:
            raise ValueError("requested charge rate exceeds maximum rate (5)")
        self.energy+=delta
        return self.energy
    
    def change(self, delta:float)->float:
        """ charge(positive value) or discharge(negative value) the battery by a given value
        assumed to be in 1 hour
        """
        if delta<0.0:
            return self.discharge(delta)
        else:
            return self.charge(delta)
        
def ensure_valid(value:float, current_consumption:float, battery:Battery)-> float:
    """ Ensure the value is within the valid range based on current demand and battery state.
    """
    if value >= current_consumption + (BATTERY_CAPCITY-battery.current_charge()):
        return current_consumption + (BATTERY_CAPCITY-battery.current_charge())
    if value >= current_consumption + BATTERY_MAX_CHARGE:
        return current_consumption + BATTERY_MAX_CHARGE
    if value <= current_consumption -battery.current_charge():
        return current_consumption - battery.current_charge()
    if value <= current_consumption -BATTERY_MAX_DISCHARGE:
        return current_consumption - BATTERY_MAX_DISCHARGE
    return value

def check_valid(value:float, current_consumption:float, battery:Battery)-> str:
    """ check if the value of consumption is within the valid range based on current consumption and battery state.
    """
    if value >= current_consumption + (BATTERY_CAPCITY-battery.current_charge()):
        return "listed battery charge rate exceeds available battery storage capacity"
    if value >= current_consumption + BATTERY_MAX_CHARGE:
        return "listed battery charge rate exceeds maximum charge rate"
    if value <= current_consumption -battery.current_charge():
        return "listed consumption exceeds available battery energy"
    if value <= current_consumption -BATTERY_MAX_DISCHARGE:
        return "listed consumption exceeds max battery discharge rate"
    return ""