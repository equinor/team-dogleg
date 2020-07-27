from flask import Flask,render_template, Markup
import matplotlib.pyplot as plt
import io
import base64
import mpld3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

app = Flask(__name__)


def plot_sphere(x0, y0, z0, r, c, ax, name=''):
    # Make data
    u = np.linspace(0, 2 * np.pi, 12)
    v = np.linspace(0, np.pi, 8)
    x = x0 + r * np.outer(np.cos(u), np.sin(v))
    y = y0 + r * np.outer(np.sin(u), np.sin(v))
    z = z0 + r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_wireframe(x, y, z, color=c)
    ax.text(x0 + r, y0 + r, z0 + r, name, None)

def plot_ball(x0,y0,z0,r,c,ax, name):    
    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = x0 + r * np.outer(np.cos(u), np.sin(v))
    y = y0 + r * np.outer(np.sin(u), np.sin(v))
    z = z0 + r * np.outer(np.ones(np.size(u)), np.cos(v))
    # Plot the surface
    ax.plot_surface(x, y, z, color=c)
    ax.text(x0+ r, y0 + r, z0 + r, name, None)



@app.route('/')
def hello_world():
    x = 5
    final = "Markup(my_html)"
    return render_template("index.html",x=str(x),stuff = final)

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