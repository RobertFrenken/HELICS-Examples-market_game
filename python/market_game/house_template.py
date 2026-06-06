"""Player template for the HELICS market game.

If you are new to HELICS and just want to play, focus on these two things:

1. Create your own file from this template, usually in `python/market_game/houses/`.
2. Implement `compute_demand()` in your subclass.

You do not need to understand the HELICS API calls in this file to play the
game. The base `House` class handles the connection, receives your 24-hour load
profile from the market maker, publishes your answer each hour, and keeps track
of cost and battery state for plotting.

What your strategy decides:
- `price`: current hour price in $/kWh
- `hour`: current hour, from 0 to 23
- `battery_charge`: energy currently stored in the battery
- `demand[hour]`: the house's base load before battery use
- `price_history`: prices seen so far, including the current price

What `compute_demand()` must return:
- the amount of power your house asks from the market this hour
- return `demand[hour]` to leave the battery unchanged
- return more than `demand[hour]` to charge the battery
- return less than `demand[hour]` to discharge the battery

Battery rules:
- maximum capacity: 20 kWh
- maximum charge in one hour: 5 kWh
- maximum discharge in one hour: 10 kWh
- initial charge: 0 kWh

If your strategy asks for an invalid amount, the template will warn you and
clamp it back into the legal range so the game can keep running.
"""

import helics as h
import json

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from battery import Battery,check_valid,ensure_valid

class House(ABC):
    """Base class for a market-game house.

    New players normally create a subclass and only override
    `compute_demand()`.
    """

    def __init__(self, name: str, connection: str = "localhost"):
        self.battery = Battery()
        self.name = name
        self.demand = [5] * 24  # Default demand profile
        self.consume = []
        self.actual_load = []
        self.prices = []
        self.actual_cost = []
        self.battery_state = []
        
        self.totalCost = 0.0
        self.current_time = 0
        self.connection = connection
        self.federate = None
        self.demand_pub = None
        self.price = None

        self.connect()

    @abstractmethod
    def compute_demand(self, price:float, hour:int, battery_charge:float, demand:list[float], price_history:list[float])->float:
        """Return the market demand for one hour.

        This is the main function players should customize.

        Sign convention:
        - `demand[hour]` means no battery action
        - larger than `demand[hour]` charges the battery
        - smaller than `demand[hour]` discharges the battery
        """
        pass
    
    def connect(self):
        fedinfo = h.helicsCreateFederateInfo()
        h.helicsFederateInfoSetCoreType(fedinfo, h.HELICS_CORE_TYPE_ZMQ_SS)
        # In most local games this stays as "localhost".
        h.helicsFederateInfoSetBroker(fedinfo, self.connection)
        h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_period, 1.0)
        
        # Create the HELICS federate for this house.
        self.federate = h.helicsCreateCombinationFederate(self.name, fedinfo)
        print(f"Created federate {self.name}")
        # Receive the hourly price from the market maker.
        self.price = self.federate.register_subscription("price", "$/kWh")
        
        # Send this house's chosen load back to the market maker.
        self.demand_pub = self.federate.register_publication("demand", "float", "kWh")
        
        # Enter initializing mode so the market maker can send our demand profile.
        self.federate.enter_initializing_mode()
        
        # Read the 24-hour demand profile assigned to this house.
        print("getting the demand profile")
        demand_profile_json = h.helicsFederateWaitCommand(self.federate)
        self.demand = json.loads(demand_profile_json)["demand"]
        
        # Enter executing mode
        self.federate.enter_executing_mode()

    def run(self):
        
        while self.current_time<24:
            # Get the current market price.
            current_price=h.helicsInputGetDouble(self.price)
            self.prices.append(current_price)
            # Base house load before battery use.
            current_demand=self.demand[int(self.current_time)]
            # Call the player strategy.
            computed_demand=self.compute_demand(current_price,int(self.current_time),self.battery.current_charge(),self.demand,self.prices)
            # Keep the strategy inside the battery limits.
            warning=check_valid(computed_demand,current_demand, self.battery)
            if warning:
                # Clamp invalid values so the house can keep running.
                print(f"invalid demand computed={computed_demand} warning={warning}, recalculating with new value")
                computed_demand=ensure_valid(computed_demand,current_demand, self.battery)
            self.actual_load.append(computed_demand)
            # Positive delta charges the battery, negative delta discharges it.
            self.battery.change(computed_demand-current_demand)
            # Save values for the summary plot.
            self.battery_state.append(self.battery.current_charge())
            self.actual_cost.append(current_price*computed_demand)
            print(f"hour {int(self.current_time)}:price={current_price}, house_demand={current_demand}, load={computed_demand}, battery delta= {computed_demand-current_demand} battery charge={self.battery.current_charge()} cost={computed_demand*current_price}")
            # Publish the decision and move to the next hour.
            self.demand_pub.publish(computed_demand)
            self.current_time=self.federate.request_next_step()
    
        print(f"total load={sum(self.actual_load)}, total_cost={sum(self.actual_cost)}")
        self.federate.disconnect()
        
    def plot_results(self):
        time = list(range(24))
        figure, axis = plt.subplots(2, 2)

        # For demand
        axis[0, 0].plot(time, self.demand, color='r', label='consumption')
        axis[0, 0].plot(time, self.actual_load, color='g', label='load')
        axis[0, 0].set_title("Demand profiles")

        # For price
        axis[0, 1].plot(time, self.prices)
        axis[0, 1].set_title("prices")

        # battery state
        axis[1, 0].plot(time, self.battery_state)
        axis[1, 0].set_title("Battery State")

        # For costs
        axis[1, 1].plot(time, self.actual_cost)
        axis[1, 1].set_title("costs")

        # Combine all the operations and display
        plt.show()

    
