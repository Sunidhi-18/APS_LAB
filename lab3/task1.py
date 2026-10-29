from lab3.data_gen import generate_nonlinear_data
from lab3.visual import plot_2d_data

def main():
    x,y = generate_nonlinear_data()
    plot_2d_data(x,y,title="nonlinearly separable data")

if __name__ == "__main__":
    main()