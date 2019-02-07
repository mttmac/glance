import random, sys
sys.path.append('../src')
from model import *

import bokeh
from bokeh.models import HoverTool, Plot, LinearAxis, Grid, Range1d
from bokeh.models.glyphs import VBar
from bokeh.models.sources import AjaxDataSource
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.models.widgets import Button, Slider
from bokeh.models.callbacks import CustomJS

from flask import Flask, render_template, jsonify, request, redirect, url_for


app = Flask(__name__)

# Load model and data in advance
model = load_checkpoint()
means, covars, threshold = load_clusters()
threshold = 0  # lower threshold for visual affect
data, targets = get_random_samples()  # input number to limit size
norm_data, fault_data = divide_samples(data, targets)
targets = 1 - targets  # invert so anomaly==1

# Initialize the monitoring cycles
n_cycles = 20
n = 0
max_n = min(norm_data.shape[0],
            fault_data.shape[0])

@app.route('/')
def index():
    sample_id = 'Hydraulic-FX5'
    return render_template("index.html",
                           sample_id=sample_id,
                           bkversion=bokeh.__version__)


@app.route('/<install_id>/')
def stream(install_id):  
    n_plots = []
    f_plots = []
    n_plots.append(create_bar_plot('cycle', 'anomaly',
                                   create_hover_tool(),
                                   method='normal'))
    f_plots.append(create_bar_plot('cycle', 'anomaly',
                                   create_hover_tool(),
                                   method='faults'))
    return render_template("dashboard.html",
                           install_id=install_id,
                           n_plots=n_plots,
                           f_plots=f_plots,
                           bkversion=bokeh.__version__)


@app.route('/normal/', methods=['POST'])
def predict_normal():
    global n
    n += 1
    if n > max_n:
        n = 1
    return predict(norm_data, np.zeros(norm_data.shape[0]))


@app.route('/faults/', methods=['POST'])
def predict_faults():
    return predict(fault_data, np.ones(norm_data.shape[0]))


@app.route('/all/', methods=['POST'])
def predict_all():
    return predict(data, targets)


def predict(data, targets):
    label = [0.01, 1]
    X = data[n - 1, :, :]
    latent, X, X_hat = compute_latent(X, model)
    log_prob = compute_log_prob(latent, means, covars)
    y_hat = detect_anomaly(log_prob, threshold)
    y = label[y_hat]
    target = targets[n - 1]
    corr = y_hat == target
    print(f"Target={target}, Prediction={y_hat}", "Correct!" if corr else "False")
    return jsonify(cycle=[n], anomaly=[y])


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
                    width=1200, height=200,
                    method='normal'):
    """
    Create a bar chart with data passed in as a dict accessible by x and y names.
    Hover tool passed in as HTML.
    """
    if method == 'normal':
        source_url = 'normal/'
    elif method == 'faults':
        source_url = 'faults/'
    else:
        source_url = 'all/'
    source = AjaxDataSource(data_url=request.url_root + source_url,
                            polling_interval=2000,
                            mode='append',
                            max_size=n_cycles)
    
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
