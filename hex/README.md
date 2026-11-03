# Running the dashboard in Hex

Hex is a notebook + app platform: analysts write SQL/Python cells, add input widgets,
and publish the result as an interactive app for stakeholders.

1. **Import**: in Hex, create a project → *Import* → upload `mmm_hex_app.ipynb`.
2. **Data**: upload `data/mmm_geo_weekly.csv` in the Files panel (in production this would be a
   SQL cell against the warehouse table that the data pipeline refreshes).
3. **Inputs**: replace the four lines in the cell marked `HEX INPUT` with Hex *Slider* inputs named
   `alpha`, `K`, `s`, `ceiling`. Every downstream cell re-runs when a slider moves.
4. **Run** the notebook once (the model cell takes a few minutes).
5. **App builder**: arrange the curve chart and the decomposition chart, hide code cells, *Publish*.

Same content as the Streamlit app in `app/dashboard.py`; Hex adds hosting, sharing,
scheduling and warehouse connections without running a server yourself.
