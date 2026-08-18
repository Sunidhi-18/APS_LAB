import matplotlib.pyplot as plt

def plot_2d_data(x,y,title="linear data"):
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.title("title")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()