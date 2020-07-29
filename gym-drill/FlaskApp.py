from flask import Flask,render_template, Markup
import matplotlib.pyplot as plt
import io
import base64
import mpld3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import sys
import agent_training as at

ACTION = sys.argv[1]
NAME = sys.argv[2]
ALGORITHM = sys.argv[3]

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)

    return base64.b64encode(img.getvalue())

app = Flask(__name__)
@app.route('/')
def hello_world():
    try:
        assert ACTION == "load"
        fig_xy,fig_xz,fig_3d = at.get_environment_figures(NAME)
        
        fig_xy_html_string = mpld3.fig_to_html(fig_xy)
        fig_xy_html_tag = Markup(fig_xy_html_string)

        fig_xz_html_string = mpld3.fig_to_html(fig_xz)
        fig_xz_html_tag = Markup(fig_xz_html_string)

        encoded = fig_to_base64(fig_3d)
        html_3d_string = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
        html_3d_tag = Markup(html_3d_string)
        return render_template("index.html",fig1 = fig_xy_html_tag,fig2 = fig_xz_html_tag,fig3 = html_3d_tag)


    except AssertionError:
        return "Invalid action specified: " + str(ACTION)


    

@app.route("/interactive")
def interactive():
    fig1,ax = plt.subplots()
    x_val = [1,2,3]
    plt.plot(x_val)

    my_html_str = mpld3.fig_to_html(fig1)
    final = Markup(my_html_str)
    # Get data
    x_positions = [1,2]
    y_positions = [1,2]
    z_positions = [1,2]
     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_positions,y_positions,z_positions)
    #ax.inINCL_zaxis()

    ax.set_xlim(1000,0)
    ax.set_ylim(0,1000)
    ax.set_zlim(1000,0)
        
    ax.set_xlabel("North")
    ax.set_ylabel("East")
    ax.set_zlabel("Down")

    three_d_string = mpld3.fig_to_html(fig)
    final2 = Markup(three_d_string)

    return render_template("index.html",x=str(5),stuff=final,morestuff=final2)
    

if __name__ == '__main__':
    print("Running on http://localhost:5000, not whatever the below message says!")
    app.run(debug=True,host="0.0.0.0")