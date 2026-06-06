# Market Game Presentation Assets

These graphics are generated from the market game's published pricing rules.

- [market_price_curve_average_demand.svg](/c:/CodeProjects/HELICS-Examples/python/market_game/presentation_assets/market_price_curve_average_demand.svg):
  main piecewise price curve versus average demand per house
- [market_price_curve_total_demand.svg](/c:/CodeProjects/HELICS-Examples/python/market_game/presentation_assets/market_price_curve_total_demand.svg):
  same rule shown as total neighborhood demand for several neighborhood sizes
- [market_price_tiers_reference.svg](/c:/CodeProjects/HELICS-Examples/python/market_game/presentation_assets/market_price_tiers_reference.svg):
  compact slide-friendly summary of the price tiers

PNG versions are included for easy drag-and-drop into presentation tools.

To regenerate the assets:

```powershell
python python/market_game/generate_presentation_graphics.py
```
