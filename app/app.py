import random
import sys
print(sys.version_info)
sys.path.append('../src')
from model import *

import bokeh
from bokeh.models import HoverTool, Plot, LinearAxis, Grid, Range1d
from bokeh.models.glyphs import VBar
from bokeh.models.sources import AjaxDataSource
from bokeh.plotting import figure
from bokeh.embed import components
from flask import Flask, render_template, jsonify, request


app = Flask(__name__)

# @before_first_request()
# def prep_data():
#     model = load_checkpoint()
#     clusters = load_clusters()

@app.route('/')
def index():
    return 'Hello World!'


@app.route('/dashboard-<install_id>/')
def stream(install_id):
    plots = []
    hover = create_hover_tool()
    plots.append(create_bar_plot('cycle', 'anomaly', hover))
    return render_template("dashboard.html",
                           install_id=install_id,
                           plots=plots,
                           bkversion=bokeh.__version__)


n_cycles = 10
x = n_cycles
label = [0.01, 1]
@app.route('/data/', methods=['POST'])
def data():
    global x
    x += 1
    y = label[random.randint(0, 1)]
    return jsonify(cycle=[x], anomaly=[y])


def create_hover_tool():
    """
    Create HTML for hover tool on bar chart.
    """
    hover_html = """
        <div>
            <span class='hover-tooltip'>Cycle: $x</span>
        </div>
        <div>
            <span class='hover-tooltip'>Anomaly: $y</span>
        </div>
        """
    return HoverTool(tooltips=hover_html)


def create_bar_plot(x_name, y_name,
                    hover_tool=None,
                    width=1200, height=200):
    """
    Create a bar chart with data passed in as a dict accessible by x and y names.
    Hover tool passed in as HTML.
    """
    print(request.url_root)
    source = AjaxDataSource(data_url=request.url_root + 'data/',
                            polling_interval=2000,
                            mode='append',
                            max_size=n_cycles)
    source.data = {x_name: list(range(1, n_cycles + 1)),
                   y_name: [0.01] * n_cycles}
    
    tools =[]
    if hover_tool:
        tools.append(hover_tool)
    plot = figure(sizing_mode='scale_width',
                  plot_width=width,
                  plot_height=height,
                  h_symmetry=False,
                  v_symmetry=False,
                  min_border=0,
                  toolbar_location='above',
                  tools=tools,
                  outline_line_color='#666666')
    plot.vbar(x=x_name,
              top=y_name,
              source=source,
              bottom=0,
              width=0.8,
              fill_color='#e12127')
    
    # plot.xaxis.ticker = source.data[x_name]
    plot.yaxis.ticker = [0, 1]
    plot.yaxis.major_label_overrides = {0: 'Normal', 1: 'Anomaly'}
    plot.toolbar.logo = None
    plot.min_border_top = 0
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = '#999999'
    plot.ygrid.grid_line_alpha = 0.1
    plot.xaxis.axis_label = 'Cycle'
    plot.xaxis.major_label_orientation = 1
    plot.y_range = Range1d(0, 1)
    plot.yaxis.major_label_text_font_size = '10pt'
    
    script, div = components(plot)
    return script, div
    

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
