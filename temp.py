import matplotlib.pyplot as plt
import io
import base64

fig,ax = plt.subplots()
x_val = [1,2,3]
plt.plot(x_val)
#plt.show()

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)

    return base64.b64encode(img.getvalue())


encoded = fig_to_base64(fig)
my_html = '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))


print("it ran")
print(my_html)